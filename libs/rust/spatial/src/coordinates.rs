//! Pointy-top axial coordinates for hexes, edges, and vertices.
//!
//! Neighbor directions run clockwise from the upper-right:
//! `0` NE, `1` E, `2` SE, `3` SW, `4` W, `5` NW.

use std::fmt;

/// Largest side length whose hexes and their immediate neighbors fit in `i8`.
pub const MAX_SPACE_SIZE: u32 = 127;

const NEIGHBORS: [(i8, i8); 6] = [
    (1, -1), // 0 NE
    (1, 0),  // 1 E
    (0, 1),  // 2 SE
    (-1, 1), // 3 SW
    (-1, 0), // 4 W
    (0, -1), // 5 NW
];

/// Axial coordinate of one hex. `q` runs along the bottom-left to top-right
/// diagonal, and a constant `q` is the top-left to bottom-right diagonal.
/// Horizontal rows share `r`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HexCoordinate {
    pub q: i8,
    pub r: i8,
}

impl HexCoordinate {
    pub const fn new(q: i8, r: i8) -> Self {
        Self { q, r }
    }

    /// Cube coordinate `s = -q - r`.
    pub const fn s(self) -> i16 {
        -(self.q as i16) - (self.r as i16)
    }

    pub const fn neighbor(self, direction: Direction) -> Self {
        let (dq, dr) = NEIGHBORS[direction.get() as usize];
        Self {
            q: self.q + dq,
            r: self.r + dr,
        }
    }

    /// A space of side length `size` contains every hex whose `q`, `r`, and `s`
    /// all lie in `[1 - size, size - 1]`.
    pub fn on_space(self, size: u32) -> bool {
        if size == 0 || size > MAX_SPACE_SIZE {
            return false;
        }
        let radius = i16::try_from(size - 1).expect("size bound checked above");
        let q = i16::from(self.q);
        let r = i16::from(self.r);
        q.abs() <= radius && r.abs() <= radius && self.s().abs() <= radius
    }
}

impl fmt::Display for HexCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.q, self.r)
    }
}

/// Clockwise index in `0..6`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Direction(u8);

impl Direction {
    pub const fn new(index: u8) -> Option<Self> {
        if index < 6 { Some(Self(index)) } else { None }
    }

    pub(crate) const fn from_index(index: u8) -> Self {
        match Self::new(index) {
            Some(direction) => direction,
            None => panic!("direction index must be in 0..6"),
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    pub const fn next(self) -> Self {
        Self((self.0 + 1) % 6)
    }

    const fn prev(self) -> Self {
        Self((self.0 + 5) % 6)
    }

    const fn plus(self, steps: u8) -> Self {
        Self((self.0 + steps) % 6)
    }
}

/// One side of a hex. `index` 0 is the upper-right side, between vertices 0 and 1,
/// and the index then increases clockwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeCoordinate {
    hex: HexCoordinate,
    index: Direction,
}

impl EdgeCoordinate {
    /// Every edge is shared by two hexes, so an edge can be referenced in two
    /// ways. The canonical name is the smaller `(q, r, d)` whose hex is on the
    /// space.
    pub fn new(hex: HexCoordinate, edge: Direction, size: u32) -> Self {
        let this = Self { hex, index: edge };
        let other = Self {
            hex: hex.neighbor(edge),
            index: edge.plus(3),
        };
        match (hex.on_space(size), other.hex.on_space(size)) {
            (true, true) => this.min(other),
            (true, false) => this,
            (false, true) => other,
            (false, false) => this.min(other),
        }
    }

    pub const fn hex(self) -> HexCoordinate {
        self.hex
    }

    pub const fn direction(self) -> Direction {
        self.index
    }
}

impl fmt::Display for EdgeCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.hex.q, self.hex.r, self.index.get())
    }
}

/// One corner of a hex. `index` 0 is the top corner, then clockwise through 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexCoordinate {
    hex: HexCoordinate,
    index: Direction,
}

impl VertexCoordinate {
    /// A vertex is shared by up to three hexes. The canonical name is the
    /// smaller `(q, r, d)` among the representations whose hex is on the space.
    pub fn new(hex: HexCoordinate, vertex: Direction, size: u32) -> Self {
        Self::aliases(hex, vertex)
            .into_iter()
            .filter(|candidate| candidate.hex.on_space(size))
            .min()
            .unwrap_or_else(|| {
                Self::aliases(hex, vertex)
                    .into_iter()
                    .min()
                    .expect("a vertex has three aliases")
            })
    }

    pub const fn hex(self) -> HexCoordinate {
        self.hex
    }

    pub const fn direction(self) -> Direction {
        self.index
    }

    /// The same corner named from this hex and from the two neighbors that meet it.
    fn aliases(hex: HexCoordinate, vertex: Direction) -> [Self; 3] {
        [
            Self { hex, index: vertex },
            Self {
                hex: hex.neighbor(vertex.prev()),
                index: vertex.plus(2),
            },
            Self {
                hex: hex.neighbor(vertex),
                index: vertex.plus(4),
            },
        ]
    }
}

impl fmt::Display for VertexCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.hex.q, self.hex.r, self.index.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direction(index: u8) -> Direction {
        Direction::new(index).expect("direction")
    }

    #[test]
    fn vertex_aliases_share_one_canonical_name() {
        let top = VertexCoordinate::new(HexCoordinate::new(0, 0), direction(0), 3);
        let from_northwest = VertexCoordinate::new(HexCoordinate::new(0, -1), direction(2), 3);
        let from_northeast = VertexCoordinate::new(HexCoordinate::new(1, -1), direction(4), 3);

        assert_eq!(top, from_northwest);
        assert_eq!(top, from_northeast);
        assert_eq!(
            (top.hex().q, top.hex().r, top.direction().get()),
            (0, -1, 2)
        );
    }

    #[test]
    fn border_vertex_keeps_an_on_space_hex() {
        let top = VertexCoordinate::new(HexCoordinate::new(0, -2), direction(0), 3);
        assert_eq!(
            (top.hex().q, top.hex().r, top.direction().get()),
            (0, -2, 0)
        );
    }

    #[test]
    fn edge_aliases_share_one_canonical_name() {
        let east = EdgeCoordinate::new(HexCoordinate::new(0, 0), direction(1), 3);
        let west_of_neighbor = EdgeCoordinate::new(HexCoordinate::new(1, 0), direction(4), 3);

        assert_eq!(east, west_of_neighbor);
        assert_eq!(east.hex(), HexCoordinate::new(0, 0));
        assert_eq!(east.direction().get(), 1);
    }

    #[test]
    fn size_three_matches_the_described_hexes() {
        let mut hexes = Vec::new();
        for q in -2..=2 {
            for r in -2..=2 {
                let hex = HexCoordinate::new(q, r);
                if hex.on_space(3) {
                    hexes.push((q, r));
                }
            }
        }

        let excluded = [(-2, -2), (-2, -1), (-1, -2), (1, 2), (2, 1), (2, 2)];
        assert_eq!(hexes.len(), 19);
        for excluded_hex in excluded {
            assert!(!hexes.contains(&excluded_hex));
        }
        assert!(hexes.contains(&(1, -1)));
        assert!(hexes.contains(&(-1, 1)));
        assert!(hexes.contains(&(1, 0)));
        assert!(hexes.contains(&(-1, 0)));
        assert!(hexes.contains(&(0, 1)));
        assert!(hexes.contains(&(0, -1)));
    }
}
