//! Decide whether a predicted path is a shortest path on the space.

use std::collections::HashMap;

use dataset::Coordinate;
use spatial::{Direction, HexCoordinate, Space, VertexCoordinate, vertex_labels};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathGrade {
    pub valid: bool,
    pub length: Option<usize>,
    pub correct: bool,
}

pub fn grade_coordinate_path(
    space: &Space,
    origin: Coordinate,
    destination: Coordinate,
    predicted: &[Coordinate],
) -> PathGrade {
    let Some(origin) = to_vertex(space, origin) else {
        return PathGrade {
            valid: false,
            length: None,
            correct: false,
        };
    };
    let Some(destination) = to_vertex(space, destination) else {
        return PathGrade {
            valid: false,
            length: None,
            correct: false,
        };
    };
    let mut vertices = Vec::with_capacity(predicted.len());
    for coordinate in predicted {
        let Some(vertex) = to_vertex(space, *coordinate) else {
            return PathGrade {
                valid: false,
                length: None,
                correct: false,
            };
        };
        vertices.push(vertex);
    }
    grade_vertices(space, origin, destination, &vertices)
}

pub fn grade_graph_path(
    space: &Space,
    origin: &str,
    destination: &str,
    predicted: &[String],
) -> PathGrade {
    let labels = vertex_labels(space);
    let mut by_label = HashMap::with_capacity(labels.len());
    for (label, vertex) in labels.iter().zip(space.vertices()) {
        by_label.insert(label.as_str(), *vertex);
    }
    let Some(&origin) = by_label.get(origin) else {
        return PathGrade {
            valid: false,
            length: None,
            correct: false,
        };
    };
    let Some(&destination) = by_label.get(destination) else {
        return PathGrade {
            valid: false,
            length: None,
            correct: false,
        };
    };
    let mut vertices = Vec::with_capacity(predicted.len());
    for label in predicted {
        let Some(&vertex) = by_label.get(label.as_str()) else {
            return PathGrade {
                valid: false,
                length: None,
                correct: false,
            };
        };
        vertices.push(vertex);
    }
    grade_vertices(space, origin, destination, &vertices)
}

fn grade_vertices(
    space: &Space,
    origin: VertexCoordinate,
    destination: VertexCoordinate,
    path: &[VertexCoordinate],
) -> PathGrade {
    let length = path.len().checked_sub(1);
    let valid = path_is_valid(space, origin, destination, path);
    let correct = valid
        && length.is_some_and(|steps| {
            space
                .distance(origin, destination)
                .is_some_and(|distance| steps == distance)
        });
    PathGrade {
        valid,
        length,
        correct,
    }
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

fn to_vertex(space: &Space, coordinate: Coordinate) -> Option<VertexCoordinate> {
    let direction = u8::try_from(coordinate.d).ok().and_then(Direction::new)?;
    Some(VertexCoordinate::new(
        HexCoordinate::new(coordinate.q, coordinate.r),
        direction,
        space.size(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coordinate(q: i8, r: i8, d: i8) -> Coordinate {
        Coordinate { q, r, d }
    }

    fn path(steps: &[(i8, i8, i8)]) -> Vec<Coordinate> {
        steps
            .iter()
            .map(|(q, r, d)| coordinate(*q, *r, *d))
            .collect()
    }

    #[test]
    fn coordinate_grade_accepts_another_shortest_path_and_rejects_a_longer_one() {
        let space = Space::new(1).expect("space");
        let origin = coordinate(0, 0, 0);
        let destination = coordinate(0, 0, 3);
        let stored = path(&[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)]);
        let alternate = path(&[(0, 0, 0), (0, 0, 5), (0, 0, 4), (0, 0, 3)]);
        let longer = path(&[
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3),
        ]);

        let exact = grade_coordinate_path(&space, origin, destination, &stored);
        assert!(exact.correct);
        assert!(exact.valid);
        assert_eq!(exact.length, Some(3));

        let other = grade_coordinate_path(&space, origin, destination, &alternate);
        assert!(other.correct);
        assert!(other.valid);
        assert_eq!(other.length, Some(3));

        let rejected = grade_coordinate_path(&space, origin, destination, &longer);
        assert!(!rejected.correct);
        assert!(rejected.valid);
        assert_eq!(rejected.length, Some(5));
    }

    #[test]
    fn valid_but_too_long_path_is_not_correct() {
        let space = Space::new(1).expect("space");
        let origin = coordinate(0, 0, 0);
        let destination = coordinate(0, 0, 3);
        let detour = path(&[
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3),
        ]);

        let grade = grade_coordinate_path(&space, origin, destination, &detour);
        assert!(grade.valid);
        assert_eq!(grade.length, Some(5));
        assert!(!grade.correct);
    }

    #[test]
    fn graph_grade_accepts_shortest_path_and_rejects_a_non_edge() {
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

        let stored = labels(&space, &[origin, neighbor, destination]);
        let broken = labels(&space, &[origin, unrelated, destination]);
        let origin_label = stored[0].clone();
        let destination_label = stored[2].clone();

        let exact = grade_graph_path(&space, &origin_label, &destination_label, &stored);
        assert!(exact.correct);
        assert!(exact.valid);

        let rejected =
            grade_graph_path(&space, &origin_label, &destination_label, &broken);
        assert!(!rejected.correct);
        assert!(!rejected.valid);
    }

    fn labels(space: &Space, vertices: &[VertexCoordinate]) -> Vec<String> {
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
