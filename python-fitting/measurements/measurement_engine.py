"""
MHR Measurement Engine
======================

PURPOSE
    Provides reusable geometry functions for measuring
    circumferences on MHR body meshes.

RESPONSIBILITIES
    - Intersect a triangle mesh with a measurement plane.
    - Merge duplicate intersection points.
    - Build and validate the intersection graph.
    - Extract closed intersection loops.
    - Project points into plane-local 2D coordinates.
    - Select the loop containing a specified point.
    - Calculate circumference from loop perimeter.

DESIGN
    Receives a mesh and a measurement plane; returns
    geometric measurement results and diagnostic information.

DOES NOT
    - Generate or modify the MHR model.
    - Define anatomical measurement locations.
    - Optimize identity coefficients.
    - Implement UI or slider interactions.

ROLE IN PROJECT
    Shared geometry layer for measurement experiments
    and future real-time avatar fitting.
"""
import numpy as np
from collections import defaultdict
import time

def edge_plane_intersection(
    a,
    b,
    plane_origin,
    plane_normal
):
    edge = b - a

    denominator = np.dot(plane_normal, edge)

    if abs(denominator) < 1e-8:
        return None

    t = np.dot(
        plane_normal,
        plane_origin - a
    ) / denominator

    if t < 0.0 or t > 1.0:
        return None

    return a + t * edge


def triangle_plane_intersections(
    v0,
    v1,
    v2,
    plane_origin,
    plane_normal
):
    edges = [
        (v0, v1),
        (v1, v2),
        (v2, v0),
    ]

    points = []

    for a, b in edges:
        point = edge_plane_intersection(
            a,
            b,
            plane_origin,
            plane_normal
        )

        if point is not None:
            points.append(point)

    # Remove duplicate intersections within one triangle
    unique = []

    for point in points:
        if not any(
            np.linalg.norm(point - p) < 1e-6
            for p in unique
        ):
            unique.append(point)

    return unique


def mesh_plane_intersections(
    vertices,
    faces,
    plane_origin,
    plane_normal
):
    segments = []

    for face in faces:

        v0, v1, v2 = vertices[face]

        points = triangle_plane_intersections(
            v0,
            v1,
            v2,
            plane_origin,
            plane_normal
        )

        if len(points) == 2:
            segments.append(
                (points[0], points[1])
            )

    return segments


def mesh_plane_intersections_slow(
    
    vertices,
    faces,
    plane_origin,
    plane_normal
):
    segments = []

    for face in faces:

        v0, v1, v2 = vertices[face]

        points = triangle_plane_intersections(
            v0,
            v1,
            v2,
            plane_origin,
            plane_normal
        )

        if len(points) == 2:
            segments.append(
                (points[0], points[1])
            )

    return segments

# Use numpy vectors for fast filtering of candidate triangles before computing intersections
# Results in a significant speedup for large meshes(x15 for 100k triangles)
def mesh_plane_intersections(
    vertices,
    faces,
    plane_origin,
    plane_normal
):
    """
    Find mesh edges intersecting a plane.

    First filters triangles using vectorized signed-distance
    calculations, then runs the existing intersection function
    only on candidate triangles.
    """
    import numpy as np

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    # Signed distance from each vertex to the plane.
    signed_distances = (
        (vertices - plane_origin) @ plane_normal
    )

    # Get signed distances for all triangle vertices at once.
    triangle_distances = signed_distances[faces]

    # A triangle can intersect the plane only if its vertices
    # lie on both sides of the plane, or at least one lies on it.
    candidate_mask = (
        (triangle_distances.min(axis=1) <= 0) &
        (triangle_distances.max(axis=1) >= 0)
    )

    candidate_faces = faces[candidate_mask]

    segments = []

    # Preserve your existing intersection logic for each candidate.
    for face in candidate_faces:
        v0, v1, v2 = vertices[face]

        points = triangle_plane_intersections(
            v0,
            v1,
            v2,
            plane_origin,
            plane_normal
        )

        if len(points) == 2:
            segments.append((points[0], points[1]))

    return segments

def build_intersection_graph(segments, tolerance=1e-3):
    """
    Build a graph from mesh-plane intersection segments.

    Spatial hashing accelerates point deduplication by checking
    only nearby grid cells instead of scanning every existing point.

    Returns:
        unique_points: NumPy array of unique 3D points
        segment_ids: List of edges represented by point indices
        adjacency: Dictionary mapping each point to its neighbors
    """
    from collections import defaultdict
    from itertools import product

    unique_points = []
    segment_ids = []

    # Map each 3D grid cell to the indices of points inside it.
    grid = defaultdict(list)

    def find_or_create_point(point):
        point = np.asarray(point, dtype=float)
        cell = tuple(np.floor(point / tolerance).astype(int))

        # A matching point may lie in any of the 26 neighboring cells
        # or in the current cell.
        for offset in product((-1, 0, 1), repeat=3):
            neighbor_cell = tuple(
                cell[axis] + offset[axis] for axis in range(3)
            )

            for index in grid[neighbor_cell]:
                if np.linalg.norm(point - unique_points[index]) < tolerance:
                    return index

        # No existing point is close enough: create a new one.
        index = len(unique_points)
        unique_points.append(point.copy())
        grid[cell].append(index)

        return index

    # Convert each segment endpoint into a graph vertex.
    for p0, p1 in segments:
        id0 = find_or_create_point(p0)
        id1 = find_or_create_point(p1)

        if id0 == id1:
            continue

        segment_ids.append((id0, id1))

    unique_points = np.asarray(unique_points, dtype=float)

    # Build the undirected graph.
    adjacency = defaultdict(list)

    for id0, id1 in segment_ids:
        adjacency[id0].append(id1)
        adjacency[id1].append(id0)

    return unique_points, segment_ids, adjacency


def extract_loops(adjacency):

    if not adjacency:
        return []

    bad_vertices = [
        point_id
        for point_id, neighbors in adjacency.items()
        if len(neighbors) != 2
    ]

    if bad_vertices:
        print(
            "\nCannot extract closed loops."
            f" {len(bad_vertices)} vertices do not have degree 2."
        )

        for point_id in bad_vertices[:20]:
            print(
                f"  point {point_id}: "
                f"degree={len(adjacency[point_id])}, "
                f"position={unique_points[point_id]}"
            )

        return []

    visited = set()
    loops = []

    for start in adjacency:

        if start in visited:
            continue

        loop = []

        previous = None
        current = start

        while True:

            loop.append(current)
            visited.add(current)

            neighbors = adjacency[current]

            next_point = (
                neighbors[0]
                if neighbors[0] != previous
                else neighbors[1]
            )

            if next_point == start:
                break

            previous = current
            current = next_point

        loops.append(loop)

    return loops


def loop_perimeter(loop, points):

    loop_points = points[loop]

    distances = np.linalg.norm(
        np.roll(loop_points, -1, axis=0)
        - loop_points,
        axis=1
    )

    return distances.sum()


def project_point_to_plane(
    point,
    plane_origin,
    plane_normal
):
    distance = np.dot(
        point - plane_origin,
        plane_normal
    )

    return point - distance * plane_normal


def point_inside_loop(
    point_2d,
    loop_2d
):
    """
    Ray-casting point-in-polygon test.
    """

    x, y = point_2d

    inside = False

    j = len(loop_2d) - 1

    for i in range(len(loop_2d)):

        xi, yi = loop_2d[i]
        xj, yj = loop_2d[j]

        intersects = (
            (yi > y) != (yj > y)
            and
            x < (
                (xj - xi)
                * (y - yi)
                / (yj - yi)
                + xi
            )
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def build_plane_basis(plane_normal):
    """
    Construct two orthonormal axes lying inside the plane.
    """

    n = plane_normal / np.linalg.norm(plane_normal)

    # Choose a reference vector that isn't parallel to n
    if abs(n[0]) < 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    else:
        reference = np.array([0.0, 1.0, 0.0])

    # First axis in the plane
    u = np.cross(n, reference)
    u /= np.linalg.norm(u)

    # Second axis in the plane
    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    return u, v


def project_points_to_plane_2d(
    points,
    plane_origin,
    plane_normal
):
    """
    Convert 3D points into 2D coordinates
    relative to the measurement plane.
    """

    u, v = build_plane_basis(plane_normal)

    relative = points - plane_origin

    x = relative @ u
    y = relative @ v

    return np.column_stack((x, y))


def find_loop_containing_point(
    loops,
    points,
    selected_point,
    plane_origin,
    plane_normal
):
    """
    Find which intersection loop contains
    the selected 3D point.
    """

    # Project selected point onto plane
    projected_point = project_point_to_plane(
        selected_point,
        plane_origin,
        plane_normal
    )

    # Convert selected point to plane-local 2D coordinates
    selected_2d = project_points_to_plane_2d(
        projected_point[None, :],
        plane_origin,
        plane_normal
    )[0]

    for loop_index, loop in enumerate(loops):

        loop_points = points[loop]

        # Convert entire loop into plane-local 2D coordinates
        loop_2d = project_points_to_plane_2d(
            loop_points,
            plane_origin,
            plane_normal
        )

        if point_inside_loop(
            selected_2d,
            loop_2d
        ):
            return loop_index

    return None


def measure_circumference(
    vertices,
    faces,
    plane_origin,
    plane_normal,
    selected_point=None,
    tolerance=1e-3,
):
    """Measure the closed mesh loop containing a selected point."""

    plane_normal = np.asarray(plane_normal, dtype=float)
    plane_normal /= np.linalg.norm(plane_normal)

    if selected_point is None:
        selected_point = plane_origin

    t0 = time.perf_counter()

    segments = mesh_plane_intersections(
        vertices, faces, plane_origin, plane_normal
    )

    print(f"  Triangle intersections: {time.perf_counter() - t0:.4f} s")

    if np.allclose(plane_normal, [0, 1, 0], atol=1e-8):
        intersection_y = np.array([
            point[1]
            for segment in segments
            for point in segment
        ])

        print(
            "Raw intersection Y range:",
            intersection_y.min(),
            intersection_y.max(),
            "range:",
            np.ptp(intersection_y),
        )

    t1 = time.perf_counter()
    points, segment_ids, adjacency = build_intersection_graph(
        segments, tolerance=tolerance
    )
    print(f"  Graph construction: {time.perf_counter() - t1:.4f} s")

    # Reject malformed graphs instead of silently returning
    # an incorrect circumference.
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None, {
            "error": "Intersection graph contains non-degree-2 vertices",
            "point_count": len(points),
        }

    t2 = time.perf_counter()
    
    loops = extract_loops(adjacency)
    print(f"  Loop extraction: {time.perf_counter() - t2:.4f} s")

    t3 = time.perf_counter()
    loop_index = find_loop_containing_point(
        loops,
        points,
        selected_point,
        plane_origin,
        plane_normal,
    )

    if loop_index is None:
        return None, {
            "error": "No loop contains the selected point",
            "point_count": len(points),
            "loop_count": len(loops),
        }

    circumference = loop_perimeter(loops[loop_index], points)
    print(f"  Loop selection + perimeter: {time.perf_counter() - t3:.4f} s")
    
    return circumference, {
    "loop_index": loop_index,
    "loop_count": len(loops),
    "point_count": len(points),
    "contour_points": points[loops[loop_index]].copy(),
    }
    
    
