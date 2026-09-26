use bytepair_core::encoder::{self, Allowed};
use bytepair_core::pretokenize::Splitter;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

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

/// Learn merges from `text` (split with `pattern`, or whole when None) up to `vocab_size`.
#[pyfunction]
#[pyo3(signature = (text, pattern, vocab_size))]
fn train(py: Python<'_>, text: &str, pattern: Option<String>, vocab_size: usize) -> PyResult<Vec<(u32, u32)>> {
    py.allow_threads(|| bytepair_core::train::train_from_text(text, pattern.as_deref(), vocab_size))
        .map_err(value_error)
}

#[pyclass(frozen, module = "bytepair._core")]
struct Encoder {
    inner: encoder::Encoder,
}

#[pymethods]
impl Encoder {
    #[staticmethod]
    #[pyo3(signature = (merges, pattern, specials))]
    fn from_merges(
        py: Python<'_>,
        merges: Vec<(u32, u32)>,
        pattern: Option<String>,
        specials: Vec<(String, u32)>,
    ) -> PyResult<Self> {
        let inner = py
            .allow_threads(|| encoder::Encoder::from_merges(&merges, pattern.as_deref(), specials))
            .map_err(value_error)?;
        Ok(Encoder { inner })
    }

    /// `ranks`: tiktoken's mergeable ranks as a list of (bytes, rank) pairs.
    #[staticmethod]
    #[pyo3(signature = (ranks, pattern, specials))]
    fn from_ranks(
        py: Python<'_>,
        ranks: Vec<(Bound<'_, PyBytes>, u32)>,
        pattern: Option<String>,
        specials: Vec<(String, u32)>,
    ) -> PyResult<Self> {
        let ranks: Vec<(Vec<u8>, u32)> = ranks.iter().map(|(b, r)| (b.as_bytes().to_vec(), *r)).collect();
        let inner = py
            .allow_threads(|| encoder::Encoder::from_ranks(ranks, pattern.as_deref(), specials))
            .map_err(value_error)?;
        Ok(Encoder { inner })
    }

    fn encode_ordinary(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        py.allow_threads(|| self.inner.encode_ordinary(text)).map_err(value_error)
    }

    /// `allowed_special`: "all", "none", "none_raise" or an iterable of special strings.
    #[pyo3(signature = (text, allowed_special=None))]
    fn encode(&self, py: Python<'_>, text: &str, allowed_special: Option<Bound<'_, PyAny>>) -> PyResult<Vec<u32>> {
        let names: Vec<String>;
        let Some(allowed_special) = allowed_special else {
            return py.allow_threads(|| self.inner.encode(text, Allowed::NoneRaise)).map_err(value_error);
        };
        let allowed = if let Ok(mode) = allowed_special.downcast::<PyString>() {
            match mode.to_cow()?.as_ref() {
                "all" => Allowed::All,
                "none" => Allowed::None,
                "none_raise" => Allowed::NoneRaise,
                other => return Err(value_error(format!("bad allowed_special: '{other}'"))),
            }
        } else {
            names = allowed_special
                .iter()
                .map_err(|_| value_error("allowed_special must be a mode string or an iterable of strings"))?
                .map(|item| item.and_then(|s| s.extract::<String>()))
                .collect::<PyResult<_>>()?;
            Allowed::Set(&names)
        };
        py.allow_threads(|| self.inner.encode(text, allowed)).map_err(value_error)
    }

    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        py.allow_threads(|| self.inner.encode_batch(&refs)).map_err(value_error)
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&ids).map_err(value_error)
    }

    fn decode_bytes<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.decode_bytes(&ids).map_err(value_error)?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    fn clear_cache(&self) {
        self.inner.clear_cache();
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", bytepair_core::VERSION)?;
    m.add("CL100K_PATTERN", bytepair_core::pretokenize::CL100K_PATTERN)?;
    m.add("O200K_PATTERN", bytepair_core::pretokenize::O200K_PATTERN)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_class::<Encoder>()?;
    Ok(())
}
