//! Random shortest-path questions in coordinate and graph form.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use spatial::{
    Space, VertexCoordinate, adjacency_list, render_coordinate_representation, vertex_labels,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetOptions {
    pub count: usize,
    pub seed: u64,
    pub coordinate: bool,
    pub graph: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coordinate {
    pub q: i8,
    pub r: i8,
    pub d: i8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathQuestionCoordinate {
    pub origin: Coordinate,
    pub destination: Coordinate,
    pub path: Vec<Coordinate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathQuestionGraph {
    pub origin: String,
    pub destination: String,
    pub path: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Question {
    pub id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinates: Option<PathQuestionCoordinate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<PathQuestionGraph>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetError {
    TooManyQuestions {
        count: usize,
        available: usize,
    },
    Unreachable {
        start: VertexCoordinate,
        end: VertexCoordinate,
    },
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyQuestions { count, available } => write!(
                f,
                "cannot sample {count} questions from {available} ordered pairs of distinct vertices"
            ),
            Self::Unreachable { start, end } => {
                write!(f, "no path from vertex {start} to vertex {end}")
            }
        }
    }
}

impl std::error::Error for DatasetError {}

#[derive(Debug)]
pub enum WriteJsonlError {
    Dataset(DatasetError),
    Io(std::io::Error),
}

impl std::fmt::Display for WriteJsonlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset(error) => write!(f, "{error}"),
            Self::Io(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for WriteJsonlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Dataset(error) => Some(error),
            Self::Io(error) => Some(error),
        }
    }
}

impl From<DatasetError> for WriteJsonlError {
    fn from(error: DatasetError) -> Self {
        Self::Dataset(error)
    }
}

impl From<std::io::Error> for WriteJsonlError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Lazily yields one [`Question`] per sampled vertex pair.
pub struct QuestionGenerator<'a> {
    space: &'a Space,
    vertices: &'a [VertexCoordinate],
    labels: Vec<String>,
    pairs: PairSampler,
    coordinate: bool,
    graph: bool,
    next_id: u32,
}

struct PairSampler {
    rng: StdRng,
    vertex_count: usize,
    remaining: usize,
    chosen: HashSet<(usize, usize)>,
}

impl Iterator for PairSampler {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        loop {
            let start = self.rng.random_range(0..self.vertex_count);
            let end = self.rng.random_range(0..self.vertex_count);
            if start != end && self.chosen.insert((start, end)) {
                self.remaining -= 1;
                return Some((start, end));
            }
        }
    }
}

impl<'a> Iterator for QuestionGenerator<'a> {
    type Item = Result<Question, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        let (start_index, end_index) = self.pairs.next()?;
        let start = self.vertices[start_index];
        let end = self.vertices[end_index];
        let path = match self.space.shortest_path(start, end) {
            Some(path) => path,
            None => {
                return Some(Err(DatasetError::Unreachable { start, end }));
            }
        };
        let id = self.next_id;
        self.next_id += 1;
        Some(Ok(Question {
            id,
            coordinates: self.coordinate.then(|| PathQuestionCoordinate {
                origin: coordinate_value(start),
                destination: coordinate_value(end),
                path: path.iter().copied().map(coordinate_value).collect(),
            }),
            graph: self.graph.then(|| PathQuestionGraph {
                origin: self.labels[start_index].clone(),
                destination: self.labels[end_index].clone(),
                path: path
                    .iter()
                    .map(|vertex| self.labels[vertex_index(self.vertices, *vertex)].clone())
                    .collect(),
            }),
        }))
    }
}

pub fn generate(
    space: &Space,
    options: DatasetOptions,
) -> Result<QuestionGenerator<'_>, DatasetError> {
    let (coordinate, graph) = match (options.coordinate, options.graph) {
        (false, false) => (true, true),
        flags => flags,
    };
    let vertices = space.vertices();
    let labels = vertex_labels(space);
    Ok(QuestionGenerator {
        space,
        vertices,
        labels,
        pairs: sample_pairs(vertices.len(), options.count, options.seed)?,
        coordinate,
        graph,
        next_id: 0,
    })
}

pub fn write_jsonl(
    path: &Path,
    questions: impl IntoIterator<Item = Result<Question, DatasetError>>,
) -> Result<usize, WriteJsonlError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    let mut written = 0;
    for question in questions {
        let question = question?;
        serde_json::to_writer(&mut file, &question).map_err(std::io::Error::other)?;
        writeln!(file)?;
        written += 1;
    }
    Ok(written)
}

pub fn write_adjacency_list(path: &Path, space: &Space) -> Result<(), std::io::Error> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let list = adjacency_list(space);
    let file = File::create(path)?;
    serde_json::to_writer_pretty(&file, &list).map_err(std::io::Error::other)?;
    Ok(())
}

pub fn write_coordinate_documentation(path: &Path, size: u32) -> Result<(), std::io::Error> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, render_coordinate_representation(size))
}

/// Standard space metadata written alongside dataset files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpaceMetadata {
    pub version: u32,
    pub space: SpaceMetadataBody,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpaceMetadataBody {
    /// Number of hexes along one side of the space.
    pub size: u32,
    pub orientation: String,
    pub hex_coordinates: String,
    pub hex_count: usize,
    pub vertex_count: usize,
}

pub fn space_metadata(space: &Space) -> SpaceMetadata {
    SpaceMetadata {
        version: 1,
        space: SpaceMetadataBody {
            size: space.size(),
            orientation: "pointy-top".to_string(),
            hex_coordinates: "axial".to_string(),
            hex_count: space.hexes().len(),
            vertex_count: space.vertices().len(),
        },
    }
}

pub fn write_metadata(path: &Path, space: &Space) -> Result<(), std::io::Error> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let metadata = space_metadata(space);
    let file = File::create(path)?;
    serde_json::to_writer_pretty(&file, &metadata).map_err(std::io::Error::other)?;
    Ok(())
}

fn sample_pairs(vertex_count: usize, count: usize, seed: u64) -> Result<PairSampler, DatasetError> {
    if vertex_count < 2 {
        if count == 0 {
            return Ok(PairSampler {
                rng: StdRng::seed_from_u64(seed),
                vertex_count,
                remaining: 0,
                chosen: HashSet::new(),
            });
        }
        return Err(DatasetError::TooManyQuestions {
            count,
            available: 0,
        });
    }

    let available = vertex_count * (vertex_count - 1);
    if count > available {
        return Err(DatasetError::TooManyQuestions { count, available });
    }

    Ok(PairSampler {
        rng: StdRng::seed_from_u64(seed),
        vertex_count,
        remaining: count,
        chosen: HashSet::with_capacity(count),
    })
}

fn coordinate_value(vertex: VertexCoordinate) -> Coordinate {
    Coordinate {
        q: vertex.hex().q,
        r: vertex.hex().r,
        d: i8::try_from(vertex.direction().get()).expect("direction fits in i8"),
    }
}

fn vertex_index(vertices: &[VertexCoordinate], vertex: VertexCoordinate) -> usize {
    vertices
        .binary_search(&vertex)
        .expect("path vertex is on the space")
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatial::{Direction, HexCoordinate};
    use std::collections::HashSet;

    fn vertex_from_coordinate(coordinate: Coordinate, size: u32) -> VertexCoordinate {
        VertexCoordinate::new(
            HexCoordinate::new(coordinate.q, coordinate.r),
            Direction::new(u8::try_from(coordinate.d).expect("direction")).expect("direction"),
            size,
        )
    }

    fn hop_distance(space: &Space, start: VertexCoordinate, goal: VertexCoordinate) -> usize {
        let mut seen = HashSet::from([start]);
        let mut layer = vec![start];
        let mut distance = 0;
        while !layer.is_empty() {
            if layer.contains(&goal) {
                return distance;
            }
            distance += 1;
            let mut next = Vec::new();
            for vertex in layer {
                for neighbor in space.neighbors(vertex) {
                    if seen.insert(*neighbor) {
                        next.push(*neighbor);
                    }
                }
            }
            layer = next;
        }
        panic!("vertex {goal} is unreachable from {start}");
    }

    #[test]
    fn generated_paths_match_bfs_distance_and_graph_labels() {
        let space = Space::new(3).expect("space");
        let questions = generate(
            &space,
            DatasetOptions {
                count: 20,
                seed: 7,
                coordinate: true,
                graph: true,
            },
        )
        .expect("questions")
        .collect::<Result<Vec<_>, _>>()
        .expect("questions");

        assert_eq!(questions.len(), 20);
        let vertices = space.vertices();
        for question in questions {
            let coordinates = question.coordinates.as_ref().expect("coordinates");
            let graph = question.graph.as_ref().expect("graph");

            assert_eq!(coordinates.path.first().copied(), Some(coordinates.origin));
            assert_eq!(
                coordinates.path.last().copied(),
                Some(coordinates.destination)
            );
            assert_eq!(
                graph.path.first().map(String::as_str),
                Some(graph.origin.as_str())
            );
            assert_eq!(
                graph.path.last().map(String::as_str),
                Some(graph.destination.as_str())
            );
            assert_eq!(coordinates.path.len(), graph.path.len());

            let path: Vec<_> = coordinates
                .path
                .iter()
                .copied()
                .map(|coordinate| vertex_from_coordinate(coordinate, 3))
                .collect();
            for step in path.windows(2) {
                assert!(space.neighbors(step[0]).contains(&step[1]));
            }
            assert_eq!(
                path.len() - 1,
                hop_distance(&space, path[0], path[path.len() - 1])
            );

            let labels = vertex_labels(&space);
            for (vertex, label) in path.iter().zip(&graph.path) {
                let index = vertex_index(vertices, *vertex);
                assert_eq!(label, &labels[index]);
            }
        }
    }

    #[test]
    fn count_above_the_number_of_pairs_is_rejected() {
        let space = Space::new(1).expect("space");
        let available = 6 * 5;
        let Err(error) = generate(
            &space,
            DatasetOptions {
                count: available + 1,
                seed: 0,
                coordinate: true,
                graph: false,
            },
        ) else {
            panic!("too many");
        };
        assert_eq!(
            error,
            DatasetError::TooManyQuestions {
                count: available + 1,
                available,
            }
        );
    }

    #[test]
    fn jsonl_round_trip_preserves_both_representations() {
        let space = Space::new(1).expect("space");
        let questions = generate(
            &space,
            DatasetOptions {
                count: 2,
                seed: 0,
                coordinate: false,
                graph: false,
            },
        )
        .expect("both representations by default")
        .collect::<Result<Vec<_>, _>>()
        .expect("questions");
        assert_eq!(questions.len(), 2);

        let path = std::env::temp_dir().join(format!("hex-dataset-{}.jsonl", std::process::id()));
        write_jsonl(&path, questions.iter().cloned().map(Ok)).expect("write");
        let contents = std::fs::read_to_string(&path).expect("read");
        let _ = std::fs::remove_file(&path);

        let parsed: Vec<Question> = contents
            .lines()
            .map(|line| serde_json::from_str(line).expect("json"))
            .collect();
        assert_eq!(parsed, questions);
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
