//! Graph-style node labels and adjacency for a space.

use std::collections::BTreeMap;

use crate::VertexCoordinate;
use crate::space::Space;

/// Adjacency list keyed by node label in canonical `(q, r, d)` order.
pub fn adjacency_list(space: &Space) -> BTreeMap<String, Vec<String>> {
    let vertices = space.vertices();
    let labels = vertex_labels(space);
    let mut list = BTreeMap::new();
    for (index, vertex) in vertices.iter().enumerate() {
        let neighbors: Vec<String> = space
            .neighbors(*vertex)
            .iter()
            .map(|neighbor| labels[vertex_index(vertices, *neighbor)].clone())
            .collect();
        list.insert(labels[index].clone(), neighbors);
    }
    list
}

/// Spreadsheet-style labels (`A`, `B`, …, `Z`, `AA`, …) in canonical vertex order.
pub fn vertex_labels(space: &Space) -> Vec<String> {
    (0..space.vertices().len()).map(column_label).collect()
}

fn column_label(index: usize) -> String {
    let mut n = index + 1;
    let mut chars = Vec::new();
    while n > 0 {
        n -= 1;
        let digit = u8::try_from(n % 26).expect("base-26 digit");
        chars.push(b'A' + digit);
        n /= 26;
    }
    chars.reverse();
    String::from_utf8(chars).expect("ascii label")
}

fn vertex_index(vertices: &[VertexCoordinate], vertex: VertexCoordinate) -> usize {
    vertices
        .binary_search(&vertex)
        .expect("vertex is on the space")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Space;

    #[test]
    fn column_labels_follow_spreadsheet_order() {
        assert_eq!(column_label(0), "A");
        assert_eq!(column_label(1), "B");
        assert_eq!(column_label(25), "Z");
        assert_eq!(column_label(26), "AA");
        assert_eq!(column_label(27), "AB");
    }

    #[test]
    fn adjacency_list_matches_space_edges() {
        let space = Space::new(1).expect("space");
        let list = adjacency_list(&space);
        let labels = vertex_labels(&space);
        assert_eq!(list.len(), labels.len());
        for (label, neighbors) in &list {
            let index = labels
                .iter()
                .position(|candidate| candidate == label)
                .expect("label");
            let vertex = space.vertices()[index];
            let expected: Vec<String> = space
                .neighbors(vertex)
                .iter()
                .map(|neighbor| labels[vertex_index(space.vertices(), *neighbor)].clone())
                .collect();
            assert_eq!(neighbors, &expected);
        }
    }
}
