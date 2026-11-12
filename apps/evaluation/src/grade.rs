//! Decide whether a predicted path is a shortest path on the space.

use dataset::Coordinate;
use spatial::{PathValidation, Space, validate_coordinate_path, validate_graph_path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathGrade {
    pub valid: bool,
    pub length: Option<usize>,
}

impl From<PathValidation> for PathGrade {
    fn from(validation: PathValidation) -> Self {
        Self {
            valid: validation.valid,
            length: validation.length,
        }
    }
}

pub fn grade_coordinate_path(
    space: &Space,
    origin: Coordinate,
    destination: Coordinate,
    predicted: &[Coordinate],
) -> PathGrade {
    grade(validate_coordinate_path(
        space,
        (origin.q, origin.r, origin.d),
        (destination.q, destination.r, destination.d),
        &predicted
            .iter()
            .map(|coordinate| (coordinate.q, coordinate.r, coordinate.d))
            .collect::<Vec<_>>(),
    ))
}

pub fn grade_graph_path(
    space: &Space,
    origin: &str,
    destination: &str,
    predicted: &[String],
) -> PathGrade {
    grade(validate_graph_path(space, origin, destination, predicted))
}

fn grade(validation: PathValidation) -> PathGrade {
    validation.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatial::{Direction, HexCoordinate, Space, VertexCoordinate, vertex_labels};

    fn coordinate(q: i8, r: i8, d: i8) -> Coordinate {
        Coordinate { q, r, d }
    }

    fn path(steps: &[(i8, i8, i8)]) -> Vec<Coordinate> {
        steps
            .iter()
            .map(|(q, r, d)| coordinate(*q, *r, *d))
            .collect()
    }

    fn to_vertex(space: &Space, coordinate: Coordinate) -> Option<VertexCoordinate> {
        let direction = u8::try_from(coordinate.d).ok().and_then(Direction::new)?;
        Some(VertexCoordinate::new(
            HexCoordinate::new(coordinate.q, coordinate.r),
            direction,
            space.size(),
        ))
    }

    fn is_optimal(
        space: &Space,
        origin: Coordinate,
        destination: Coordinate,
        grade: PathGrade,
    ) -> bool {
        grade.valid
            && grade.length.is_some_and(|steps| {
                let Some(origin) = to_vertex(space, origin) else {
                    return false;
                };
                let Some(destination) = to_vertex(space, destination) else {
                    return false;
                };
                space
                    .distance(origin, destination)
                    .is_some_and(|distance| steps == distance)
            })
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
        assert!(is_optimal(&space, origin, destination, exact));
        assert!(exact.valid);
        assert_eq!(exact.length, Some(3));

        let other = grade_coordinate_path(&space, origin, destination, &alternate);
        assert!(is_optimal(&space, origin, destination, other));
        assert!(other.valid);
        assert_eq!(other.length, Some(3));

        let rejected = grade_coordinate_path(&space, origin, destination, &longer);
        assert!(!is_optimal(&space, origin, destination, rejected));
        assert!(rejected.valid);
        assert_eq!(rejected.length, Some(5));
    }

    #[test]
    fn valid_but_too_long_path_is_not_optimal() {
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
        assert!(!is_optimal(&space, origin, destination, grade));
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
        assert!(exact.valid);
        assert_eq!(exact.length, Some(2));

        let rejected = grade_graph_path(&space, &origin_label, &destination_label, &broken);
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
