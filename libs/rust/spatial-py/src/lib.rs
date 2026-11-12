use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use spatial::{
    Direction, HexCoordinate, PathValidation, Space, VertexCoordinate,
    render_coordinate_representation, render_graph_representation, validate_coordinate_path,
    validate_graph_path,
};

#[pyclass(name = "Space")]
struct PySpace {
    inner: Space,
}

#[pymethods]
impl PySpace {
    #[new]
    fn new(size: u32) -> PyResult<Self> {
        Space::new(size)
            .map(|inner| Self { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[getter]
    fn size(&self) -> u32 {
        self.inner.size()
    }

    fn neighbors(&self, q: i8, r: i8, d: i8) -> PyResult<Vec<(i8, i8, i8)>> {
        let vertex = vertex_from_parts(q, r, d, self.inner.size())?;
        Ok(self
            .inner
            .neighbors(vertex)
            .iter()
            .map(vertex_to_tuple)
            .collect())
    }

    fn distance(&self, a: (i8, i8, i8), b: (i8, i8, i8)) -> PyResult<Option<usize>> {
        let start = vertex_from_parts(a.0, a.1, a.2, self.inner.size())?;
        let end = vertex_from_parts(b.0, b.1, b.2, self.inner.size())?;
        Ok(self.inner.distance(start, end))
    }

    fn render_graph_representation(&self) -> String {
        render_graph_representation(&self.inner)
    }
}

#[pyfunction]
fn render_coordinate(size: u32) -> String {
    render_coordinate_representation(size)
}

#[pyfunction]
fn render_graph(size: u32) -> PyResult<String> {
    let space = Space::new(size).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(render_graph_representation(&space))
}

#[pyclass(name = "PathValidation")]
#[derive(Clone, Copy)]
struct PyPathValidation {
    valid: bool,
    length: Option<usize>,
}

#[pymethods]
impl PyPathValidation {
    #[getter]
    fn valid(&self) -> bool {
        self.valid
    }

    #[getter]
    fn length(&self) -> Option<usize> {
        self.length
    }
}

impl From<PathValidation> for PyPathValidation {
    fn from(validation: PathValidation) -> Self {
        Self {
            valid: validation.valid,
            length: validation.length,
        }
    }
}

#[pyfunction]
fn py_validate_coordinate_path(
    size: u32,
    origin: (i8, i8, i8),
    destination: (i8, i8, i8),
    path: Vec<(i8, i8, i8)>,
) -> PyResult<PyPathValidation> {
    let space = Space::new(size).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(validate_coordinate_path(&space, origin, destination, &path).into())
}

#[pyfunction]
fn py_validate_graph_path(
    size: u32,
    origin: &str,
    destination: &str,
    path: Vec<String>,
) -> PyResult<PyPathValidation> {
    let space = Space::new(size).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(validate_graph_path(&space, origin, destination, &path).into())
}

fn vertex_from_parts(q: i8, r: i8, d: i8, size: u32) -> PyResult<VertexCoordinate> {
    let direction_index = u8::try_from(d)
        .map_err(|_| PyValueError::new_err(format!("vertex direction must be in 0..6, got {d}")))?;
    let direction = Direction::new(direction_index).ok_or_else(|| {
        PyValueError::new_err(format!("vertex direction must be in 0..6, got {d}"))
    })?;
    Ok(VertexCoordinate::new(
        HexCoordinate::new(q, r),
        direction,
        size,
    ))
}

fn vertex_to_tuple(vertex: &VertexCoordinate) -> (i8, i8, i8) {
    (
        vertex.hex().q,
        vertex.hex().r,
        i8::try_from(vertex.direction().get()).expect("direction fits in i8"),
    )
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpace>()?;
    m.add_class::<PyPathValidation>()?;
    m.add_function(wrap_pyfunction!(render_coordinate, m)?)?;
    m.add_function(wrap_pyfunction!(render_graph, m)?)?;
    m.add_function(wrap_pyfunction!(py_validate_coordinate_path, m)?)?;
    m.add_function(wrap_pyfunction!(py_validate_graph_path, m)?)?;
    Ok(())
}
