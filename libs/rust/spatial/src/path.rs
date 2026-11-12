//! Path validation on the vertex graph.

use std::collections::HashMap;

use crate::coordinates::{Direction, HexCoordinate, VertexCoordinate};
use crate::graph::vertex_labels;
use crate::space::Space;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathValidation {
    pub valid: bool,
    pub length: Option<usize>,
}

impl PathValidation {
    const INVALID: Self = Self {
        valid: false,
        length: None,
    };
}

/// Check whether `path` is a walk from `origin` to `destination` along graph edges.
pub fn validate_path(
    space: &Space,
    origin: VertexCoordinate,
    destination: VertexCoordinate,
    path: &[VertexCoordinate],
) -> PathValidation {
    let length = path.len().checked_sub(1);
    let valid = path_is_valid(space, origin, destination, path);
    PathValidation { valid, length }
}

pub fn validate_coordinate_path(
    space: &Space,
    origin: (i8, i8, i8),
    destination: (i8, i8, i8),
    path: &[(i8, i8, i8)],
) -> PathValidation {
    let Some(origin) = to_vertex(space, origin) else {
        return PathValidation::INVALID;
    };
    let Some(destination) = to_vertex(space, destination) else {
        return PathValidation::INVALID;
    };
    let mut vertices = Vec::with_capacity(path.len());
    for coordinate in path {
        let Some(vertex) = to_vertex(space, *coordinate) else {
            return PathValidation::INVALID;
        };
        vertices.push(vertex);
    }
    validate_path(space, origin, destination, &vertices)
}

pub fn validate_graph_path(
    space: &Space,
    origin: &str,
    destination: &str,
    path: &[String],
) -> PathValidation {
    let labels = vertex_labels(space);
    let mut by_label = HashMap::with_capacity(labels.len());
    for (label, vertex) in labels.iter().zip(space.vertices()) {
        by_label.insert(label.as_str(), *vertex);
    }
    let Some(&origin) = by_label.get(origin) else {
        return PathValidation::INVALID;
    };
    let Some(&destination) = by_label.get(destination) else {
        return PathValidation::INVALID;
    };
    let mut vertices = Vec::with_capacity(path.len());
    for label in path {
        let Some(&vertex) = by_label.get(label.as_str()) else {
            return PathValidation::INVALID;
        };
        vertices.push(vertex);
    }
    validate_path(space, origin, destination, &vertices)
}

fn path_is_valid(
    space: &Space,
    origin: VertexCoordinate,
    destination: VertexCoordinate,
    path: &[VertexCoordinate],
) -> bool {
    let (Some(&first), Some(&last)) = (path.first(), path.last()) else {
        return false;
    };
    if first != origin || last != destination {
        return false;
    }
    path.windows(2)
        .all(|step| space.neighbors(step[0]).contains(&step[1]))
}

fn to_vertex(space: &Space, coordinate: (i8, i8, i8)) -> Option<VertexCoordinate> {
    let direction = u8::try_from(coordinate.2).ok().and_then(Direction::new)?;
    Some(VertexCoordinate::new(
        HexCoordinate::new(coordinate.0, coordinate.1),
        direction,
        space.size(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Space;

    #[test]
    fn validate_path_accepts_adjacent_walk_and_rejects_bad_endpoints() {
        let space = Space::new(1).expect("space");
        let origin = space.vertices()[0];
        let neighbor = space.neighbors(origin)[0];
        let destination = space
            .neighbors(neighbor)
            .iter()
            .copied()
            .find(|vertex| *vertex != origin)
            .expect("second hop");

        let valid = validate_path(
            &space,
            origin,
            destination,
            &[origin, neighbor, destination],
        );
        assert!(valid.valid);
        assert_eq!(valid.length, Some(2));

        let wrong_end = validate_path(&space, origin, destination, &[origin, neighbor]);
        assert!(!wrong_end.valid);

        let wrong_start = validate_path(&space, origin, destination, &[neighbor, destination]);
        assert!(!wrong_start.valid);
    }

    #[test]
    fn coordinate_validation_matches_vertex_rules() {
        let space = Space::new(1).expect("space");
        let origin = (0, 0, 0);
        let destination = (0, 0, 3);
        let stored = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)];
        let longer = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3),
        ];

        let exact = validate_coordinate_path(&space, origin, destination, &stored);
        assert!(exact.valid);
        assert_eq!(exact.length, Some(3));

        let detour = validate_coordinate_path(&space, origin, destination, &longer);
        assert!(detour.valid);
        assert_eq!(detour.length, Some(5));
    }

    #[test]
    fn graph_validation_rejects_a_non_edge() {
        let space = Space::new(1).expect("space");
        let origin = space.vertices()[0];
        let neighbor = space.neighbors(origin)[0];
        let destination = space
            .neighbors(neighbor)
            .iter()
            .copied()
            .find(|vertex| *vertex != origin)
            .expect("second hop");
        let unrelated = space
            .vertices()
            .iter()
            .copied()
            .find(|vertex| {
                *vertex != origin
                    && *vertex != destination
                    && !space.neighbors(origin).contains(vertex)
            })
            .expect("non-neighbor");

        let labels = labels_for(&space, &[origin, neighbor, destination]);
        let broken = labels_for(&space, &[origin, unrelated, destination]);

        let exact = validate_graph_path(&space, &labels[0], &labels[2], &labels);
        assert!(exact.valid);
        assert_eq!(exact.length, Some(2));

        let rejected = validate_graph_path(&space, &labels[0], &labels[2], &broken);
        assert!(!rejected.valid);
    }

    fn labels_for(space: &Space, vertices: &[VertexCoordinate]) -> Vec<String> {
        let names = vertex_labels(space);
        vertices
            .iter()
            .map(|vertex| {
                let index = space
                    .vertices()
                    .iter()
                    .position(|candidate| candidate == vertex)
                    .expect("vertex");
                names[index].clone()
            })
            .collect()
    }
}
