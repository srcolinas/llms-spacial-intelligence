//! String representations of the coordinate and graph approaches.

use crate::coordinates::{Direction, HexCoordinate, VertexCoordinate};
use crate::graph::adjacency_list;
use crate::space::Space;

/// Markdown description of the coordinate system for a space of the given size.
pub fn render_coordinate_representation(size: u32) -> String {
    let min = 1 - i64::from(size);
    let max = i64::from(size) - 1;
    let radius = size - 1;
    format!(
        "# Hex space coordinates

The space is a pointy-top hexagonal patch of side length {size}, centered at hex `(0, 0)`.

## Hex coordinates

Hexes use axial coordinates `(q, r)`. The center hex is `(0, 0)`.

- `q` increases toward the bottom-right.
- `r` increases toward the right (horizontal rows).

A hex `(q, r)` is on the space when `q`, `r`, and `s = -q - r` all lie in `[{min}, {max}]`.
Equivalently, each of `|q|`, `|r|`, and `|s|` is at most `{radius}`.

## Vertex coordinates

Vertices are corners of on-space hexes. Each vertex is named `(q, r, d)`:

- `(q, r)` is the anchor hex for the name.
- `d` is the corner index on that hex: `0` at the top corner, then increasing clockwise through `5`:

```
       *d=0*
    d=5*****d=1
    d=4*****d=2
       *d=3*
```

## Equivalent names and canonical form

The same physical corner has up to three names `(q, r, d)`:

1. `(q, r, d)` on hex `(q, r)`.
2. On the hex reached from `(q, r)` in direction `(d + 5) mod 6`, name `(d + 2) mod 6`.
3. On the hex reached from `(q, r)` in direction `d`, name `(d + 4) mod 6`.

Hex neighbor offsets (axial, clockwise from upper-right): \
`0→(1,-1), 1→(1,0), 2→(0,1), 3→(-1,1), 4→(-1,0), 5→(0,-1)`.

The **canonical** name is the smallest `(q, r, d)` among aliases whose hex is on the space.
Questions and answers use canonical names. For example, on a size-3 space the top corner of hex `(0, 0)` is `(0, -1, 2)`, not `(0, 0, 0)`.

## Graph and adjacency

The graph has one node per vertex and one edge between corners that share a side of a hex.
Each vertex has degree 2 along the hex boundary.

**Neighbor rule.** From vertex `(q, r, d)`, the two neighbors are the canonical names of:

- `(q, r, (d + 1) mod 6)`
- `(q, r, (d + 5) mod 6)`

Opposite corners of the same hex are not adjacent (they are three steps apart around the hex).

Shortest-path questions ask for a minimum-length path along these edges, including both endpoints.

## Example (size 1)

{example}
"
    ,
        example = size_one_example()
    )
}

fn size_one_example() -> String {
    let space = Space::new(1).expect("size-1 space");
    let start = VertexCoordinate::new(HexCoordinate::new(0, 0), Direction::from_index(0), 1);
    let end = VertexCoordinate::new(HexCoordinate::new(0, 0), Direction::from_index(3), 1);
    let path = space
        .shortest_path(start, end)
        .expect("path on size-1 space");
    let steps: Vec<String> = path.iter().map(|vertex| vertex.to_string()).collect();
    let neighbors = space.neighbors(start);
    let neighbor_names: Vec<String> = neighbors.iter().map(|vertex| vertex.to_string()).collect();

    format!(
        "Hex `(0, 0)` has corners `(0, 0, 0)` through `(0, 0, 5)` clockwise from the top.\n\
         Neighbors of `(0, 0, 0)`: {}.\n\
         Shortest path `(0, 0, 0)` → `(0, 0, 3)`: {} (length 3).",
        neighbor_names.join(" and "),
        steps.join(", ")
    )
}

/// Human-readable adjacency list for the graph approach.
pub fn render_graph_representation(space: &Space) -> String {
    let list = adjacency_list(space);
    let mut lines = vec![format!("# Graph representation (size {})", space.size())];
    lines.push(String::new());
    lines.push(
        "Each node is labeled with a spreadsheet-style name. Neighbors are listed in canonical order."
            .to_string(),
    );
    lines.push(String::new());
    for (node, neighbors) in list {
        lines.push(format!("{node}: {}", neighbors.join(", ")));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Space;

    #[test]
    fn coordinate_representation_mentions_space() {
        let text = render_coordinate_representation(3);
        assert!(text.contains("Hex space coordinates"));
        assert!(!text.contains("board"));
    }

    #[test]
    fn coordinate_representation_uses_cube_constraint() {
        let text = render_coordinate_representation(3);
        assert!(text.contains("s = -q - r"));
        assert!(text.contains("Neighbor rule"));
        assert!(text.contains("Example (size 1)"));
    }

    #[test]
    fn graph_representation_lists_neighbors() {
        let space = Space::new(1).expect("space");
        let text = render_graph_representation(&space);
        assert!(text.contains("A:"));
        assert!(text.contains("Graph representation"));
    }
}
