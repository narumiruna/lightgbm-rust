//! Checked errors for user input, model data, and unsupported capabilities.

use std::fmt;

/// An error returned before invalid data reaches a numerical kernel.
#[derive(Debug)]
pub enum Error {
    /// The parameter or input does not satisfy the API contract.
    InvalidInput(String),
    /// The model text is malformed or internally inconsistent.
    InvalidModel(String),
    /// A recognized capability is not implemented.
    Unsupported(String),
    /// A file operation failed.
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::InvalidModel(message) => write!(f, "invalid model: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported: {message}"),
            Self::Io(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// The result type used by checked public operations.
pub type Result<T> = std::result::Result<T, Error>;
