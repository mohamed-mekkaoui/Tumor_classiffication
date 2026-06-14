import openslide
import numpy as np
import networkx as nx
import random
import ast
import geopandas as gpd
from shapely.geometry import Point

import config


def _extract_label(classification):
    """Extrait le nom de classe d'une valeur 'classification' GeoJSON.

    Gère les formes : dict {'name': ...}, str repr d'un dict, ou NaN/None.
    Retourne 'background' si rien d'exploitable.
    """
    if classification is None or isinstance(classification, float):
        return "background"
    if isinstance(classification, dict):
        return classification.get('name', 'background')
    if isinstance(classification, str):
        try:
            return ast.literal_eval(classification).get('name', 'background')
        except Exception:
            return classification
    return "background"


class WSIHexGraph:
    def __init__(self, wsi_path, patch_size=224, white_threshold=220, white_ratio=0.5, start_col=0, start_row=0, max_cols=None, max_rows=None):
        self.wsi_path = wsi_path
        self.slide = openslide.OpenSlide(wsi_path)
        self.patch_size = patch_size      # output size (model input, stored patch)
        self.out_size = patch_size
        self.white_threshold = white_threshold
        self.white_ratio = white_ratio

        self.width, self.height = self.slide.dimensions
        self.start_col = start_col
        self.start_row = start_row

        # ── Grossissement : on lit au niveau 0 une tuile `tile_l0` puis on la
        # redimensionne à `out_size` pour atteindre TARGET_MPP (20×). Les lames
        # 40× (mpp≈0.25) donnent scale=2 → tile_l0=448. Une lame déjà en 20×
        # donne scale=1 → tile_l0=224 (comportement inchangé).
        mpp_x = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        try:
            self.mpp = float(mpp_x)
        except (TypeError, ValueError):
            self.mpp = config.DEFAULT_SLIDE_MPP
        self.scale = max(1, round(config.TARGET_MPP / self.mpp))
        # Level-0 tile size used for BOTH grid stepping and patch reading.
        self.tile_l0 = self.patch_size * self.scale
        print(f"Native mpp={self.mpp:.4f} → scale x{self.scale} "
              f"(tile_l0={self.tile_l0} px @ L0 → {self.out_size} px @ {config.TARGET_MPP} mpp)")

        total_cols = self.width // self.tile_l0
        total_rows = self.height // self.tile_l0

        self.cols = min(total_cols - start_col, max_cols) if max_cols else total_cols - start_col
        self.rows = min(total_rows - start_row, max_rows) if max_rows else total_rows - start_row

        # To create a hexagonal brick-wall tiling with squares,
        # we shift odd columns down by half the (level-0) tile size.
        self.half_patch = self.tile_l0 // 2

        self.graph = nx.Graph()
        # Allows instant mapping from grid index -> graph Node id
        self.grid_nodes = {}
    

    def _get_neighbors(self, c, r):
        """
        Returns the list of 6 theoretical valid neighbors for a given patch index in a brick-wall grid.
        Where odd columns are shifted down by half patch size.
        Returns the neighbors in a strictly Clockwise order starting from Top (0->Top, 1->TR, 2->BR, 3->Bot, 4->BL, 5->TL).
        """
        if c % 2 == 0:
            return [
                (c, r - 1),     # 0: Top
                (c + 1, r - 1), # 1: Top-Right
                (c + 1, r),     # 2: Bottom-Right
                (c, r + 1),     # 3: Bottom
                (c - 1, r),     # 4: Bottom-Left
                (c - 1, r - 1)  # 5: Top-Left
            ]
        else:
            return [
                (c, r - 1),     # 0: Top
                (c + 1, r),     # 1: Top-Right
                (c + 1, r + 1), # 2: Bottom-Right
                (c, r + 1),     # 3: Bottom
                (c - 1, r + 1), # 4: Bottom-Left
                (c - 1, r)      # 5: Top-Left
            ]

    def build_graph(self):
        """
        Iterates over the entire slide, filters patches, and builds the valid Nodes and Edges
        """
        print(f"Slide dimensions: {self.width}x{self.height} -> Max Patches: {self.cols} x {self.rows}")
        
        node_id = 0
        
        # 1. Build Nodes efficiently using a downsampled region of interest mask
        print("Generating low-res tissue mask to accelerate patch filtering...")
        downsample_factor = 32
        # Use the slide's built-in pyramid level (pre-computed in SVS)
        best_level = self.slide.get_best_level_for_downsample(downsample_factor)
        actual_downsample = self.slide.level_downsamples[best_level]

        # Only read the region of interest (our cols/rows window), not the whole slide
        roi_x0 = self.start_col * self.tile_l0
        roi_y0 = self.start_row * self.tile_l0
        roi_w  = self.cols * self.tile_l0
        roi_h  = self.rows * self.tile_l0 + self.half_patch  # +half_patch for odd-col shift

        # Clamp to slide bounds
        roi_w = min(roi_w, self.width  - roi_x0)
        roi_h = min(roi_h, self.height - roi_y0)

        thumb_w = max(1, int(roi_w / actual_downsample))
        thumb_h = max(1, int(roi_h / actual_downsample))

        print(f"Using pyramid level {best_level} (downsample x{actual_downsample:.0f}), ROI thumb: {thumb_w}x{thumb_h}")
        region = self.slide.read_region((roi_x0, roi_y0), best_level, (thumb_w, thumb_h)).convert("RGB")
        thumb_array = np.array(region)
        # thumb_width/thumb_height used only for clamping below
        thumb_width, thumb_height = thumb_w, thumb_h
        
        # Calculate grayscale mask on the thumbnail
        gray_thumb = np.dot(thumb_array[...,:3], [0.2989, 0.5870, 0.1140])
        mask = gray_thumb <= self.white_threshold # True means it's tissue (not white)
        
        for c_offset in range(self.cols):
            c = self.start_col + c_offset
            x = c * self.tile_l0
            for r_offset in range(self.rows):
                r = self.start_row + r_offset
                y = r * self.tile_l0

                # Shift odd columns down
                if c % 2 == 1:
                    y += self.half_patch

                # Prevent bound overflow
                if y + self.tile_l0 > self.height or x + self.tile_l0 > self.width:
                    continue

                # Fast Check: Check corresponding region in the thumbnail mask
                # Map patch bounding box to thumbnail coordinates (relative to ROI origin)
                thumb_x1 = int((x - roi_x0) / actual_downsample)
                thumb_y1 = int((y - roi_y0) / actual_downsample)
                thumb_x2 = int((x - roi_x0 + self.tile_l0) / actual_downsample)
                thumb_y2 = int((y - roi_y0 + self.tile_l0) / actual_downsample)
                
                # Ensure within thumb bounds
                thumb_x2 = min(thumb_x2, thumb_width)
                thumb_y2 = min(thumb_y2, thumb_height)
                
                if thumb_x1 >= thumb_x2 or thumb_y1 >= thumb_y2:
                    continue
                    
                # Extract the small mask region for this patch
                patch_mask = mask[thumb_y1:thumb_y2, thumb_x1:thumb_x2]
                
                # If more than (1 - white_ratio) of the patch is solid tissue, accept it
                # Equivalently, if less than white_ratio is white.
                tissue_ratio = np.sum(patch_mask) / patch_mask.size
                
                # We want white_ratio < 0.5, which means tissue_ratio > 0.5
                if tissue_ratio > (1.0 - self.white_ratio):
                    center_x = x + self.half_patch
                    center_y = y + self.half_patch
                    
                    self.graph.add_node(node_id, 
                                        c=c, r=r, 
                                        px=x, py=y,
                                        cx=center_x, cy=center_y)
                    self.grid_nodes[(c, r)] = node_id
                    node_id += 1
            
            if (c_offset + 1) % 50 == 0 or c_offset == self.cols - 1:
                print(f"Processed column offset {c_offset + 1}/{self.cols} (Absolute Col {c})...")

        # 2. Build Edges
        print("Nodes generated. Connecting valid neighbors...")
        for (c, r), nid in self.grid_nodes.items():
            possible_neighbors = self._get_neighbors(c, r)
            for nc, nr in possible_neighbors:
                if (nc, nr) in self.grid_nodes:
                    neighbor_id = self.grid_nodes[(nc, nr)]
                    # Graph is undirected; adding an edge multiple times is ignored by NetworkX safely
                    self.graph.add_edge(nid, neighbor_id)
                    
        print(f"Graph Construction Complete: {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Edges")

    def spatial_to_node(self, x, y):
        """
        Converts a spatial Cartesian coordinate (x,y) to a Graph node.
        Returns the node ID if it falls into a valid patch, else None.
        """
        c = int(x // self.tile_l0)

        # Compute row considering the column shift
        if c % 2 == 0:
            r = int(y // self.tile_l0)
        else:
            r = int((y - self.half_patch) // self.tile_l0)

        if (c, r) in self.grid_nodes:
            return self.grid_nodes[(c, r)]
        return None

    def chess_to_brick_mapping(self, chess_c, chess_r):
        """
        Maps standard chess grid index (used by Dino raw predictions) to this staggered grid.
        This provides a quick way to pool/map standard model outputs to the valid nodes.
        """
        cx = chess_c * self.tile_l0 + self.half_patch
        cy = chess_r * self.tile_l0 + self.half_patch

        return self.spatial_to_node(cx, cy)

    def load_annotations(self, geojson_path):
        """
        Loads the JSON annotation files into a fast spatial r-tree index via GeoPandas.
        """
        print(f"Loading annotations from {geojson_path}")
        self.gdf = gpd.read_file(geojson_path)

    def tag_nodes_with_annotations(self):
        """
        Assigns an annotation class to each graph node via point-in-polygon.

        Les annotations sont EMBOÎTÉES : un gros polygone grossier ("Tumor")
        contient des polygones de sous-types fins ("ACINAIRE", ...). Un nœud
        tombe donc dans PLUSIEURS polygones à la fois.

        Règle de résolution (priorité) :
          1. Un sous-type FIN l'emporte TOUJOURS sur un label grossier
             (Tumor / no_Tumor / no_Tissu / background), car le but est de
             classifier le sous-type, pas tumor/pas-tumor.
          2. Entre deux sous-types fins (rare, régions quasi-disjointes),
             le plus PETIT polygone gagne (le plus spécifique).

        Les labels grossiers sont ceux de ``config.EXCLUDED_LABELS``.
        """
        import config

        if not hasattr(self, 'gdf'):
            print("Error: No annotations loaded. Call load_annotations() first.")
            return

        print("Tagging nodes with JSON annotations (sous-type prioritaire sur grossier)...")

        # Par défaut tous les nœuds sont 'background' (cas sans match)
        for nid in self.graph.nodes():
            self.graph.nodes[nid]['label'] = "background"

        nodes_data = []
        for nid, data in self.graph.nodes(data=True):
            nodes_data.append({
                'node_id': nid,
                'geometry': Point(data['cx'], data['cy'])
            })

        nodes_gdf = gpd.GeoDataFrame(nodes_data, crs=self.gdf.crs)

        # Aire de chaque polygone (départage entre sous-types : plus petit = plus spécifique)
        poly_area = self.gdf.geometry.area

        # Jointure spatiale : un nœud peut matcher plusieurs polygones
        joined = gpd.sjoin(nodes_gdf, self.gdf, how='inner', predicate='within')
        if len(joined) == 0:
            print("Finished tagging nodes (aucun match).")
            return

        joined = joined.copy()
        joined['poly_area'] = poly_area.loc[joined['index_right']].values
        joined['cls'] = joined['classification'].apply(_extract_label)

        # Labels grossiers (faible priorité) → perdent face aux sous-types fins
        coarse = set(getattr(config, 'EXCLUDED_LABELS',
                             ['background', 'no_Tissu', 'no_Tumor', 'Tumor']))
        joined['is_coarse'] = joined['cls'].isin(coarse)

        # Statistique de chevauchement (info diagnostique)
        multi = (joined.groupby('node_id').size() > 1).sum()

        # Pour chaque nœud : trier par (sous-type d'abord, puis plus petite aire)
        joined = joined.sort_values(['is_coarse', 'poly_area'],
                                    ascending=[True, True])
        winners = joined.groupby('node_id', sort=False).first()

        for nid, row in winners.iterrows():
            self.graph.nodes[nid]['label'] = row['cls']

        n_bg = self.graph.number_of_nodes() - len(winners)
        print(f"Finished tagging nodes : {len(winners)} taggés, {n_bg} background, "
              f"{int(multi)} nœuds en chevauchement (sous-type prioritaire).")

    def generate_random_walk(self, start_node, min_length=30, max_length=50, constraint_label=None, sharp_turn_weight=0.0):
        """
        Generates a non-backtracking random walk through the tissue graph.
        
        Args:
            start_node (int): The ID of the node to start from.
            min_length (int): Minimum number of nodes in the sequence.
            max_length (int): Maximum number of nodes in the sequence.
            constraint_label (str): If provided, walk will stay entirely within this annotation class.
            sharp_turn_weight (float): 0.0 means strict forward-only (3 out of 6 neighbors). 
                                       > 0.0 applies a probability penalty to paths that turn sharply next to the origin.
            
        Returns:
            list: A sequence of Node IDs representing the path.
        """
        if start_node not in self.graph:
            return []
            
        length = random.randint(min_length, max_length)
        walk = [start_node]
        current_node = start_node
        previous_node = None
        
        for _ in range(length - 1):
            neighbors = list(self.graph.neighbors(current_node))
            
            # Constraint 1: "Bounce" logic inside valid annotation areas
            if constraint_label is not None:
                valid_neighbors = []
                for n in neighbors:
                    if self.graph.nodes[n].get('label') == constraint_label:
                        valid_neighbors.append(n)
                neighbors = valid_neighbors
            
            if previous_node is not None:
                c_curr = self.graph.nodes[current_node]['c']
                r_curr = self.graph.nodes[current_node]['r']
                c_prev = self.graph.nodes[previous_node]['c']
                r_prev = self.graph.nodes[previous_node]['r']
                
                # Fetch theoretically perfect 6 clockwise neighbors
                cw_neighbors = self._get_neighbors(c_curr, r_curr)
                
                try:
                    prev_idx = cw_neighbors.index((c_prev, r_prev))
                except ValueError:
                    break # Failsafe
                    
                sharp_left_idx = (prev_idx - 1) % 6
                sharp_right_idx = (prev_idx + 1) % 6
                
                valid_directional_neighbors = []
                weights = []
                
                for i, (nc, nr) in enumerate(cw_neighbors):
                    if i == prev_idx:
                        continue # strict no 180 backtrack
                        
                    # Lookup graph node ID for this theoretical neighbor
                    n = self.grid_nodes.get((nc, nr))
                    if n is None or n not in neighbors:
                        continue # Node doesn't exist (edge of tissue) or isn't a valid allowed neighbor
                        
                    if i == sharp_left_idx or i == sharp_right_idx:
                        # Sharp turn back towards origin
                        if sharp_turn_weight > 0.0:
                            valid_directional_neighbors.append(n)
                            weights.append(sharp_turn_weight)
                    else:
                        # Forward, or shallow +60 / -60 turn
                        valid_directional_neighbors.append(n)
                        weights.append(1.0)
                        
                if not valid_directional_neighbors:
                    break
                    
                next_node = random.choices(valid_directional_neighbors, weights=weights, k=1)[0]
            else:
                # First step, no previous node, pick any valid neighbor uniformly
                if not neighbors:
                    break
                next_node = random.choice(neighbors)
                
            walk.append(next_node)
            previous_node = current_node
            current_node = next_node
            
        return walk

    def generate_random_walk_bounce(self, start_node, min_length=30, max_length=50, constraint_label=None, sharp_turn_weight=0.0):
        """
        Random walk with bounce logic for region-constrained generation.

        Same as ``generate_random_walk`` but when the walk reaches the
        boundary of a constrained region, it **bounces** instead of stopping:

        1. Try forward / shallow turns (normal)
        2. If stuck → allow sharp turns (±120°) at full weight
        3. If still stuck → allow backtrack (180°)
        4. Only stops if truly isolated (no neighbor at all in the region)

        This maximises walk length inside small regions.
        """
        if start_node not in self.graph:
            return []

        length = random.randint(min_length, max_length)
        walk = [start_node]
        current_node = start_node
        previous_node = None

        for _ in range(length - 1):
            neighbors = list(self.graph.neighbors(current_node))

            # Filter to constraint label
            if constraint_label is not None:
                neighbors = [
                    n for n in neighbors
                    if self.graph.nodes[n].get('label') == constraint_label
                ]

            if previous_node is not None:
                c_curr = self.graph.nodes[current_node]['c']
                r_curr = self.graph.nodes[current_node]['r']
                c_prev = self.graph.nodes[previous_node]['c']
                r_prev = self.graph.nodes[previous_node]['r']

                cw_neighbors = self._get_neighbors(c_curr, r_curr)

                try:
                    prev_idx = cw_neighbors.index((c_prev, r_prev))
                except ValueError:
                    break

                sharp_left_idx = (prev_idx - 1) % 6
                sharp_right_idx = (prev_idx + 1) % 6

                # --- Phase 1: forward + shallow turns (normal) ---
                valid_directional_neighbors = []
                weights = []

                for i, (nc, nr) in enumerate(cw_neighbors):
                    if i == prev_idx:
                        continue
                    n = self.grid_nodes.get((nc, nr))
                    if n is None or n not in neighbors:
                        continue
                    if i == sharp_left_idx or i == sharp_right_idx:
                        if sharp_turn_weight > 0.0:
                            valid_directional_neighbors.append(n)
                            weights.append(sharp_turn_weight)
                    else:
                        valid_directional_neighbors.append(n)
                        weights.append(1.0)

                # --- Phase 2: BOUNCE — sharp turns at full weight ---
                if not valid_directional_neighbors:
                    for i, (nc, nr) in enumerate(cw_neighbors):
                        if i == prev_idx:
                            continue
                        n = self.grid_nodes.get((nc, nr))
                        if n is None or n not in neighbors:
                            continue
                        valid_directional_neighbors.append(n)
                        weights.append(1.0)

                # --- Phase 3: BACKTRACK — allow 180° as last resort ---
                if not valid_directional_neighbors:
                    back_pos = cw_neighbors[prev_idx]
                    n = self.grid_nodes.get(back_pos)
                    if n is not None and n in neighbors:
                        valid_directional_neighbors.append(n)
                        weights.append(1.0)

                # Truly stuck (isolated) — stop
                if not valid_directional_neighbors:
                    break

                next_node = random.choices(valid_directional_neighbors, weights=weights, k=1)[0]
            else:
                if not neighbors:
                    break
                next_node = random.choice(neighbors)

            walk.append(next_node)
            previous_node = current_node
            current_node = next_node

        return walk
