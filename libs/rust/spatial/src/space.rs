//! Hex space and shortest paths along its vertices.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;

use crate::coordinates::{Direction, HexCoordinate, MAX_SPACE_SIZE, VertexCoordinate};

/// A pointy-top space whose side is `size` hexes long.
#[derive(Debug, Clone)]
pub struct Space {
    size: u32,
    hexes: BTreeSet<HexCoordinate>,
    vertices: Vec<VertexCoordinate>,
    adjacency: HashMap<VertexCoordinate, Vec<VertexCoordinate>>,
}

impl Space {
    pub fn new(size: u32) -> Result<Self, SpaceError> {
        if size == 0 || size > MAX_SPACE_SIZE {
            return Err(SpaceError::SizeOutOfRange { size });
        }

        let radius = i8::try_from(size - 1).expect("size bound checked above");
        let mut hexes = BTreeSet::new();
        for q in -radius..=radius {
            for r in -radius..=radius {
                let hex = HexCoordinate::new(q, r);
                if hex.on_space(size) {
                    hexes.insert(hex);
                }
            }
        }

        let mut vertices = BTreeSet::new();
        let mut adjacency: BTreeMap<VertexCoordinate, BTreeSet<VertexCoordinate>> = BTreeMap::new();
        for hex in &hexes {
            for index in 0..6 {
                let direction = Direction::from_index(index);
                let start = VertexCoordinate::new(*hex, direction, size);
                let end = VertexCoordinate::new(*hex, direction.next(), size);
                vertices.insert(start);
                vertices.insert(end);
                if start != end {
                    adjacency.entry(start).or_default().insert(end);
                    adjacency.entry(end).or_default().insert(start);
                }
            }
        }

        let vertices: Vec<VertexCoordinate> = vertices.into_iter().collect();
        let adjacency = adjacency
            .into_iter()
            .map(|(vertex, neighbors)| (vertex, neighbors.into_iter().collect()))
            .collect();

        Ok(Self {
            size,
            hexes,
            vertices,
            adjacency,
        })
    }

    pub const fn size(&self) -> u32 {
        self.size
    }

    pub const fn hexes(&self) -> &BTreeSet<HexCoordinate> {
        &self.hexes
    }

    pub fn vertices(&self) -> &[VertexCoordinate] {
        &self.vertices
    }

    /// Neighbors of `vertex` reachable in one step along the space graph.
    pub fn neighbors(&self, vertex: VertexCoordinate) -> &[VertexCoordinate] {
        self.adjacency
            .get(&vertex)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Shortest hop count between two vertices, or `None` when either vertex is
    /// off the space or no path exists.
    pub fn distance(&self, start: VertexCoordinate, end: VertexCoordinate) -> Option<usize> {
        self.shortest_path(start, end)
            .map(|path| path.len().saturating_sub(1))
    }

    /// Shortest vertex sequence from `start` to `end`, including both ends.
    ///
    /// When several paths have the same length, neighbors are explored in
    /// canonical `(q, r, d)` order, so the result is deterministic.
    pub fn shortest_path(
        &self,
        start: VertexCoordinate,
        end: VertexCoordinate,
    ) -> Option<Vec<VertexCoordinate>> {
        if !self.adjacency.contains_key(&start) || !self.adjacency.contains_key(&end) {
            return None;
        }
        if start == end {
            return Some(vec![start]);
        }

        let mut previous = HashMap::new();
        let mut queue = VecDeque::from([start]);
        while let Some(current) = queue.pop_front() {
            for &neighbor in &self.adjacency[&current] {
                if neighbor == start || previous.contains_key(&neighbor) {
                    continue;
                }
                previous.insert(neighbor, current);
                if neighbor == end {
                    let mut path = vec![end];
                    let mut cursor = end;
                    while cursor != start {
                        cursor = previous[&cursor];
                        path.push(cursor);
                    }
                    path.reverse();
                    return Some(path);
                }
                queue.push_back(neighbor);
            }
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpaceError {
    SizeOutOfRange { size: u32 },
}

impl fmt::Display for SpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOutOfRange { size } => write!(
                f,
                "space size must be from 1 to {MAX_SPACE_SIZE}, got {size}"
            ),
        }
    }
}

impl std::error::Error for SpaceError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(q: i8, r: i8, d: u8, size: u32) -> VertexCoordinate {
        VertexCoordinate::new(
            HexCoordinate::new(q, r),
            Direction::new(d).expect("direction"),
            size,
        )
    }

    #[test]
    fn size_three_has_nineteen_hexes_and_fifty_four_vertices() {
        let space = Space::new(3).expect("space");
        let excluded = [(-2, -2), (-2, -1), (-1, -2), (1, 2), (2, 1), (2, 2)];

        assert_eq!(space.hexes().len(), 19);
        assert_eq!(space.vertices().len(), 54);
        for (q, r) in excluded {
            assert!(!space.hexes().contains(&HexCoordinate::new(q, r)));
        }
        for q in -2..=2 {
            for r in -2..=2 {
                let hex = HexCoordinate::new(q, r);
                assert_eq!(space.hexes().contains(&hex), hex.on_space(3));
            }
        }

        let degree_sum: usize = space
            .vertices()
            .iter()
            .map(|vertex| space.neighbors(*vertex).len())
            .sum();
        assert_eq!(degree_sum, 144);

        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::from([space.vertices()[0]]);
        seen.insert(space.vertices()[0]);
        while let Some(current) = queue.pop_front() {
            for neighbor in space.neighbors(current) {
                if seen.insert(*neighbor) {
                    queue.push_back(*neighbor);
                }
            }
        }
        assert_eq!(seen.len(), space.vertices().len());
    }

    #[test]
    fn opposite_corners_of_one_hex_follow_the_clockwise_path() {
        let space = Space::new(1).expect("space");
        let path = space
            .shortest_path(vertex(0, 0, 0, 1), vertex(0, 0, 3, 1))
            .expect("path");
        let expected: Vec<_> = (0..4).map(|d| vertex(0, 0, d, 1)).collect();
        assert_eq!(path, expected);
        assert_eq!(
            space.distance(vertex(0, 0, 0, 1), vertex(0, 0, 3, 1)),
            Some(3)
        );
    }

    #[test]
    fn adjacent_corners_are_one_step_apart() {
        let space = Space::new(3).expect("space");
        let start = vertex(0, 0, 0, 3);
        let end = vertex(0, 0, 1, 3);
        assert_eq!(
            space.shortest_path(start, end).expect("path"),
            vec![start, end]
        );
        assert_eq!(space.distance(start, end), Some(1));
    }

    #[test]
    fn size_zero_is_rejected() {
        assert!(matches!(
            Space::new(0),
            Err(SpaceError::SizeOutOfRange { size: 0 })
        ));
    }
}
