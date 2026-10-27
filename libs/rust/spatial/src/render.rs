//! String representations of the coordinate and graph approaches.

use crate::graph::adjacency_list;
use crate::space::Space;

/// Markdown description of the coordinate system for a space of the given size.
pub fn render_coordinate_representation(size: u32) -> String {
    let min = 1 - i64::from(size);
    let max = i64::from(size) - 1;
    format!(
        "# Hex space coordinates

The space is a pointy-top hexagonal grid of side length {size}.

## Hex coordinates

Hexes use axial coordinates `(q, r)`. The center hex is `(0, 0)`.

- `q` runs along the bottom-left to top-right diagonal.
- `r` runs horizontally across rows.

A hex is on the space when `q` and `r` all lie in `[{min}, {max}]`.

## Vertex coordinates

Vertices are named `(q, r, d)`:

- `(q, r)` is the hex the name is anchored to.
- `d` is the corner index on that hex: `0` at the top corner, then increasing clockwise through `5`.

Each corner is shared by up to three hexes, so it has several equivalent names. The name used in this dataset is the **canonical** one: among the representations whose hex is on the space, we pick the smallest `(q, r, d)` in numerical order. All vertices are converted to canonical form when evaluated.

## Adjacency

Two vertices are connected when they are adjacent corners of the same hex. Shortest-path questions ask for a path that moves only along those connections.
"
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
    fn graph_representation_lists_neighbors() {
        let space = Space::new(1).expect("space");
        let text = render_graph_representation(&space);
        assert!(text.contains("A:"));
        assert!(text.contains("Graph representation"));
    }
}
