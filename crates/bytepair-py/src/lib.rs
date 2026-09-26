use bytepair_core::pretokenize::Splitter;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Split `text` the way `regex.findall(pattern, text)` does. Known OpenAI patterns use
/// the hand-written matchers unless `force_regex` is set.
#[pyfunction]
#[pyo3(signature = (pattern, text, force_regex=false))]
fn split(pattern: &str, text: &str, force_regex: bool) -> PyResult<Vec<String>> {
    let splitter = if force_regex {
        Splitter::regex(pattern)
    } else {
        Splitter::for_pattern(Some(pattern))
    }
    .map_err(value_error)?;
    let pieces = splitter.split(text).map_err(value_error)?;
    Ok(pieces.into_iter().map(String::from).collect())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", bytepair_core::VERSION)?;
    m.add("CL100K_PATTERN", bytepair_core::pretokenize::CL100K_PATTERN)?;
    m.add("O200K_PATTERN", bytepair_core::pretokenize::O200K_PATTERN)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    Ok(())
}
