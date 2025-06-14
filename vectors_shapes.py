import fitz
import numpy as np
import math
from collections import defaultdict
from itertools import permutations, combinations

def points_equal(p1, p2, tol=1e-6):
    return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

def canonical_point(pt, tol=1e-6):
    # Round to avoid floating point errors
    return (round(pt[0]/tol)*tol, round(pt[1]/tol)*tol)

def build_graph_with_elements(lines, tol=1e-6):
    point_id = {}
    id_point = []
    def get_id(pt):
        cpt = canonical_point(pt, tol)
        if cpt not in point_id:
            point_id[cpt] = len(id_point)
            id_point.append(cpt)
        return point_id[cpt]
    
    edges = {}
    for line in lines:
        u = get_id(line['coords'][0])
        v = get_id(line['coords'][1])
        if u != v:
            # Store the line element as the value for the edge
            edges[tuple(sorted((u, v)))] = line
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges.keys():
        adj[u].add(v)
        adj[v].add(u)
    return adj, id_point, edges

def find_triangles_with_elements(elements, tol=1e-6):
    # 1. Extract line elements
    lines = [el for el in elements if el['type'] == 'Line']
    # 2. Build graph with elements
    adj, id_point, edges = build_graph_with_elements(lines, tol)
    triangles = set()
    # 3. For each node, check pairs of neighbors
    for u in adj:
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i+1, len(neighbors)):
                w = neighbors[j]
                # If v and w are connected, we have a triangle
                if w in adj[v]:
                    # Sort node indices to avoid duplicates
                    tri_nodes = tuple(sorted((u, v, w)))
                    triangles.add(tri_nodes)
    # 4. Convert back to elements
    triangle_elements = []
    for tri in triangles:
        u, v, w = tri
        # Get the three edges (lines) that form the triangle
        edge1 = edges[tuple(sorted((u, v)))]
        edge2 = edges[tuple(sorted((v, w)))]
        edge3 = edges[tuple(sorted((w, u)))]
        if edge1 is None or edge2 is None or edge3 is None:
            continue
        triangle_elements.append((edge1, edge2, edge3))
    return triangle_elements

"""
# triangles alternate method

# def points_equal(p1, p2, tol=1e-6):
#     return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

# def find_triangles(elements):
#     # Extract only the line elements
#     lines = [el['coords'] for el in elements if el['type'] == 'Line']
#     triangles = []
#     for combo in combinations(lines, 3):
#         # Gather all endpoints
#         points = []
#         for line in combo:
#             points.extend(line)
#         # Find unique points
#         unique_points = []
#         for pt in points:
#             if not any(points_equal(pt, up) for up in unique_points):
#                 unique_points.append(pt)
#         # A triangle must have exactly 3 unique points
#         if len(unique_points) != 3:
#             continue
#         # Check if each line connects two unique points
#         valid = True
#         for line in combo:
#             if not (any(points_equal(line[0], up) for up in unique_points) and
#                     any(points_equal(line[1], up) for up in unique_points)):
#                 valid = False
#                 break
#         if not valid:
#             continue
#         # Check if each point is an endpoint of exactly two lines
#         point_line_count = {tuple(pt): 0 for pt in unique_points}
#         for line in combo:
#             for pt in unique_points:
#                 if points_equal(line[0], pt):
#                     point_line_count[tuple(pt)] += 1
#                 if points_equal(line[1], pt):
#                     point_line_count[tuple(pt)] += 1
#         if all(count == 2 for count in point_line_count.values()):
#             triangles.append(combo)
#     return triangles
"""

def bezier_point(t, control_points):
    """Calculate point on Bézier curve using De Casteljau's algorithm - O(n) time"""
    points = np.array(control_points)
    n = len(points) - 1
    
    # Use binomial coefficients for direct calculation - more efficient
    result = np.zeros(2)
    for i in range(n + 1):
        binomial_coeff = math.comb(n, i)
        bernstein = binomial_coeff * (t ** i) * ((1 - t) ** (n - i))
        result += bernstein * points[i]
    
    return tuple(result)

def sample_bezier_points(control_points, num_samples=20):
    """Sample points along Bézier curve - O(n*m) where n=control points, m=samples"""
    return [bezier_point(t, control_points) for t in np.linspace(0, 1, num_samples)]

def estimate_circle_center_fast(points):
    """Fast circle center estimation using algebraic method - O(n) time"""
    points = np.array(points)
    n = len(points)
    
    if n < 3:
        return None, None
    
    # Use algebraic circle fitting (faster than least squares)
    x = points[:, 0]
    y = points[:, 1]
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Center coordinates using simplified algebraic method
    u = x - x_mean
    v = y - y_mean
    
    # Avoid division by zero
    denominator = 2 * (np.sum(u * u) + np.sum(v * v))
    if abs(denominator) < 1e-10:
        return x_mean, y_mean
    
    uc = np.sum(u * u * u + u * v * v) / denominator
    vc = np.sum(v * v * v + v * u * u) / denominator
    
    center_x = uc + x_mean
    center_y = vc + y_mean
    
    return center_x, center_y

def is_single_bezier_circle(control_points, circle_tolerance=1, min_radius=1):
    """
    Optimized circle detection for single Bézier curve - O(n) time
    """
    if len(control_points) < 3:
        return False, None, None, None
    
    # Quick check: if start and end points are far apart, not a closed curve
    start = np.array(control_points[0])
    end = np.array(control_points[-1])
    dist_start_end = np.linalg.norm(end - start)

    if dist_start_end > circle_tolerance * 20:  # Adjusted threshold
        return False, None, None, None

    # Adaptive sampling based on curve complexity
    num_control_points = len(control_points)
    num_samples = min(30, max(12, num_control_points * 3))
    
    # Sample points efficiently
    points = sample_bezier_points(control_points, num_samples)
    points_array = np.array(points)
    
    # Fast center estimation
    cx, cy = estimate_circle_center_fast(points)
    if cx is None:
        return False, None, None, None
    
    # Vectorized distance calculation
    center = np.array([cx, cy])
    distances = np.linalg.norm(points_array - center, axis=1)
    avg_radius = np.mean(distances)
    
    if avg_radius < min_radius:
        return False, None, None, None
    
    # Check radius consistency using vectorized operations
    relative_deviations = np.abs(distances - avg_radius) / avg_radius
    max_deviation = np.max(relative_deviations)
    
    if max_deviation > circle_tolerance:
        return False, None, None, None
    
    # Fast aspect ratio check using min/max
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    
    if height == 0 or width == 0:
        return False, None, None, None
    
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 1.2:
        return False, None, None, None
    
    return True, (cx, cy), avg_radius, max_deviation

def find_quarter_circles_optimized(elements):
    """
    Find circles from Bézier elements - O(k*n) where k=number of elements, n=control points
    """
    circles = []
    
    for el in elements:
        if el['type'] == 'Bezier':
            control_points = el['coords']
            
            # Skip if too few control points
            if len(control_points) < 3:
                continue
                
            is_circle, center, radius, max_deviation = is_single_bezier_circle(control_points)
            if is_circle:
                circles.append({
                    'element': el,
                    'center': center,
                    'radius': radius,
                    'max_deviation': max_deviation,
                    'control_points': len(control_points)
                })
    
    return circles

def build_connectivity_graph(bezier_elements, tol=1e-3):
    """Build adjacency graph based on shared endpoints"""
    adjacency = defaultdict(list)
    for i, el1 in enumerate(bezier_elements):
        for j, el2 in enumerate(bezier_elements):
            if i != j:
                # Check if end of el1 connects to start of el2
                if points_equal(el1['element']['coords'][-1], el2['element']['coords'][0], tol):
                    adjacency[i].append(j)
    
    return adjacency, bezier_elements

def find_closed_chains(adjacency, chain_length=4):
    """
    Find all simple cycles (closed chains) of specified length in an undirected graph.
    adjacency: dict mapping node -> set/list of neighbors
    chain_length: desired cycle length (e.g., 4 for quarter-circle detection)
    Returns: list of cycles, each as a list of node indices
    """
    chains = set()

    def dfs(path, visited):
        current = path[-1]
        # If we've reached the desired length, check if it forms a cycle
        if len(path) == chain_length:
            if path[0] in adjacency[current]:
                # Normalize cycle to avoid duplicates (start from smallest node)
                cycle = tuple(sorted(path))
                chains.add(cycle)
            return
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                dfs(path + [neighbor], visited | {neighbor})

    nodes = list(adjacency.keys())
    for start in nodes:
        dfs([start], {start})

    # Remove duplicate cycles (cycles with same set of nodes)
    unique_chains = []
    seen = set()
    for cycle in chains:
        key = tuple(sorted(cycle))
        if key not in seen:
            seen.add(key)
            # Reconstruct the cycle in the correct order
            # Find a path in the adjacency graph that visits all nodes in 'cycle' in order
            # For most applications, the sorted tuple is sufficient as a representative
            unique_chains.append(list(cycle))
    return unique_chains

def detect_circles_from_connected_beziers(elements, min_radius=5):
    """Main algorithm to detect circles from connected Bézier elements"""

    quarter_circles= find_quarter_circles_optimized(elements)    
    
    # Build connectivity graph
    adjacency, bezier_elements = build_connectivity_graph(quarter_circles)
    # Find closed chains of length 4 (typical for circle approximation)
    chains = find_closed_chains(adjacency, chain_length=4)
    circles = []
    for chain in chains:
        circle=[]
        for idx in chain:
            circle.append(bezier_elements[idx])
        circles.append(circle)    
    return circles

def find_5_pointed_stars_with_elements(elements, tol=1e-6):
    # 1. Extract line elements
    lines = [el for el in elements if el['type'] == 'Line']
    # 2. Build graph with elements
    adj, id_point, edges = build_graph_with_elements(lines, tol)
    stars = []
    visited_cycles = set()
    
    # 3. Find cycles of length 10 (5-pointed star has 10 edges)
    def find_star_cycles(start_node, current_path, visited_nodes):
        if len(current_path) == 10:
            # Check if we can close the cycle back to start
            current_node = current_path[-1]
            if start_node in adj[current_node]:
                # Normalize cycle to avoid duplicates
                cycle_key = tuple(sorted(current_path))
                if cycle_key not in visited_cycles:
                    visited_cycles.add(cycle_key)
                    return [current_path]
            return []
        
        results = []
        current_node = current_path[-1]
        for neighbor in adj[current_node]:
            if neighbor not in visited_nodes:
                new_path = current_path + [neighbor]
                new_visited = visited_nodes | {neighbor}
                results.extend(find_star_cycles(start_node, new_path, new_visited))
        return results
    
    # 4. Search for star cycles starting from each node
    for start_node in adj:
        if len(adj[start_node]) >= 2:  # Star points need at least 2 connections
            cycles = find_star_cycles(start_node, [start_node], {start_node})
            for cycle in cycles:
                # Convert node cycle to line elements
                star_elements = []
                valid_star = True
                for i in range(10):
                    u = cycle[i]
                    v = cycle[(i + 1) % 10]
                    edge_key = tuple(sorted((u, v)))
                    if edge_key in edges:
                        star_elements.append(edges[edge_key])
                    else:
                        valid_star = False
                        break
                
                if valid_star and len(star_elements) == 10:
                    # Validate star geometry
                    if is_valid_star_geometry(star_elements, id_point, cycle):
                        stars.append(star_elements)
    
    return stars

def is_valid_star_geometry(star_elements, id_point, node_cycle):
    """Validate 5-pointed star using alternating convex/concave angles"""
    points = [id_point[node] for node in node_cycle]
    
    # Calculate internal angles at each vertex
    angles = []
    for i in range(len(points)):
        prev = np.array(points[i-1])
        current = np.array(points[i])
        next_p = np.array(points[(i+1)%len(points)])
        
        # Create vectors
        v1 = prev - current
        v2 = next_p - current
        
        # Calculate angle between vectors (0-360 degrees)
        angle = math.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
        angle = (angle + 360) % 360  # Normalize to 0-360
        
        # Store internal angle (complement if reflex)
        internal_angle = angle if angle < 180 else 360 - angle
        angles.append(internal_angle if angle < 180 else 360 - internal_angle)
    # print(f"here: {min(angles)}")
    if min(angles)>50:
        
        angles = [360-angle for angle in angles]
    # Check alternating pattern (convex < 180°, concave > 180°)
    convex_count = 0
    concave_count = 0
    pattern_valid = True
    
    for i in range(len(angles)):
        current = angles[i]
        next_angle = angles[(i+1)%len(angles)]
        
        # Check alternation between convex and concave
        if (current < 180 and next_angle < 180) or \
           (current > 180 and next_angle > 180):
            pattern_valid = False
            break
        
        if current < 180:
            convex_count += 1
        else:
            concave_count += 1
    # Validate counts and sum (5 convex tips @ ~36° each = 180° total)
    convex_sum = sum(a for a in angles if a < 180)
    # print(convex_sum)
    return (
        pattern_valid and 
        convex_count == 5 and 
        concave_count == 5 and
        170 < convex_sum < 190  # Allow 10° tolerance
    )


PDF_PATH = "hp_naina_devi.pdf"
PAGE_NUM = 0  # zero-based

doc = fitz.open(PDF_PATH)
page = doc[PAGE_NUM]    
def extract_drawings(page, zoom=1, page_height=None):
    drawings = page.get_drawings()
    elements = []
    for path in drawings:
        color = path.get('color')
        if color is None:
            color_hex = '#000000'  # default to black
        else:
            # Convert floats (0.0–1.0) to ints (0–255) if necessary
            if all(isinstance(c, float) for c in color):
                color_ints = tuple(int(c * 255) for c in color)
            else:
                color_ints = color
            color_hex = '#{:02x}{:02x}{:02x}'.format(*color_ints)


        width = path.get('width') or 1
        for item in path["items"]:
            cmd = item[0]
            if cmd == 'l':  # Line
                p1, p2 = item[1], item[2]
                coords = [p1, p2]
                typ = "Line"
            elif cmd == 're':  # Rectangle
                rect = item[1]
                coords = [rect.tl, rect.tr, rect.br, rect.bl, rect.tl]
                typ = "Rectangle"
            elif cmd == 'c':  # Bezier curve
                p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
                coords = [p1, p2, p3, p4]
                typ = "Bezier"
            elif cmd == 'qu':  # Quad
                quad = item[1]
                coords = [quad.ul, quad.ur, quad.lr, quad.ll, quad.ul]
                typ = "Quad"
            else:
                continue  # skip unsupported
            # Transform PDF coordinates to image coordinates (PDF: bottom-left, Image: top-left)
            coords_img = [
                (
                    pt.x * zoom,
                    pt.y * zoom
                ) for pt in coords
            ]
            elements.append({
                'type': typ,
                'coords': coords_img,
                'info': f"Type: {typ}\nCoords: {coords}\nColor: {color_hex}\nWidth: {width}",
                'color': color_hex,
                'width': width
            })
    return elements
elements = extract_drawings(page)

triangles = find_triangles_with_elements(elements)
circles = detect_circles_from_connected_beziers(elements)
stars = find_5_pointed_stars_with_elements(elements)
# Print results
# for i,circle in enumerate(circles):
#     print(f"Circle {i+1}: {circle}")
    # print(f"Element Info: {circle['element']['info']}")
    # print(f"Control Points: {circle['element']['coords']}\n")
# print(f"Total Stars Found: {len(triangles)}")
# print(f"Total Circles Found: {len(circles)}")
# print(f"Total Stars Found: {len(stars)}")
def element_points_equal_list(list1, list2, tol=1e-6):
    """Check if two lists of points are equal (forward or reverse), within tolerance."""
    if len(list1) != len(list2):
        return False
    # Forward comparison
    if all(points_equal(a, b, tol) for a, b in zip(list1, list2)):
        return True
    # Reverse comparison
    if all(points_equal(a, b, tol) for a, b in zip(list1, reversed(list2))):
        return True
    return False

def elements_equal(el1, el2, tol=1e-6):
    """Check if two elements (line or bezier) are the same, order-insensitive."""
    if el1['type'] != el2['type']:
        return False
    if el1['type'] == 'Line':
        # Each line has two endpoints
        return element_points_equal_list(el1['coords'], el2['coords'], tol)
    elif el1['type'] == 'Bezier':
        # Bézier: compare all control points (order-insensitive)
        return element_points_equal_list(el1['coords'], el2['coords'], tol)
    # Extend for other types as needed
    return False

filtered_elements = []
for el in elements:
    in_shape = False
    for tri in triangles:
        if any(elements_equal(el, tri_el) for tri_el in tri):
            in_shape = True
            break
    if not in_shape:
        for circle in circles:
            if any(elements_equal(el, circ_el['element']) for circ_el in circle):
                in_shape = True
                break
    if not in_shape:
        for star in stars:
            if any(elements_equal(el, star_el) for star_el in star):
                in_shape = True
                break
    if in_shape:
        filtered_elements.append(el)
print("done")









from PIL import Image
import io
import base64
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

PDF_PATH = "hp_naina_devi.pdf"
PAGE_NUM = 0  # zero-based

# 1. Render the PDF page as an image
def render_page_as_image(doc, page_num, zoom=1):
    page = doc[page_num]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return img_b64, pix.width, pix.height, zoom

doc = fitz.open(PDF_PATH)
img_b64, img_w, img_h, zoom = render_page_as_image(doc, PAGE_NUM)
page = doc[PAGE_NUM]

# 4. Build Plotly figure
fig = go.Figure()

# Add background image
fig.add_layout_image(
    dict(
        source=f"data:image/png;base64,{img_b64}",
        xref="x", yref="y",
        x=0, y=0,
        sizex=img_w, sizey=img_h,
        sizing="stretch",
        opacity=1,
        layer="below"
    )
)

# Add vector overlays
for el in filtered_elements:
    # if not el['type'] == 'Rectangle':
    #     continue
    x = [pt[0] for pt in el['coords']]
    y = [pt[1] for pt in el['coords']]
    mode = 'lines+markers' if el['type'] in ['Line', 'Bezier', 'Quad'] else 'lines'
    extra_info = ""
    if 'radius' in el and 'center' in el:
        extra_info = f"Radius: {el['radius']:.4f} Center: ({el['center'][0]:.4f}, {el['center'][1]:.4f})"
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode=mode,
        text=[el['info']+extra_info] * len(x),
        hoverinfo='text',
        name=el['type'],
        line=dict(color=el['color'], width=max(3, el['width'] * zoom)),
        marker=dict(size=6, color=el['color'])
    ))

fig.update_layout(
    title="PDF Vector Graphics Visualization",
    hovermode='closest',
    xaxis=dict(visible=False, range=[0, img_w]),
    yaxis=dict(visible=False, range=[img_h, 0]),
    margin=dict(l=0, r=0, t=40, b=0),
    width=img_w,
    height=img_h
)

# Dash app
app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='pdf-graph', figure=fig, config={'displayModeBar': False}),
    html.Div(id='element-info', style={'whiteSpace': 'pre-line', 'marginTop': '10px'})
])

@app.callback(
    Output('element-info', 'children'),
    Input('pdf-graph', 'clickData')
)
def display_element_info(clickData):
    if clickData:
        point = clickData['points'][0]
        return f"Details:\n{point['text']}"
    return "Click on an element to see details."

if __name__ == '__main__':
    app.run(debug=True)