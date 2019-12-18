#![feature(try_trait)]

use std::collections::BTreeMap;
use std::ops::Try;
use serde::{ Serialize, Deserialize };

#[derive(PartialEq)]
pub enum DataObtainError {
    // Peeking
    NonePeek, // peeking None
    ConversionError(String, String), // Cant convert using peek

    StringParseError(String, String), // Error parsing `String` to `Type`

    // Object
    OtherAsObject(String), // Called obj_get on Object type
    NoSuchKey(String), // Map doesn't contain key

    // Array
    OtherAsArray(String), // Called arr_get on non Array type
    ArrayIndexOutOfBounds(usize, usize),

}
impl std::error::Error for DataObtainError {}
impl std::fmt::Debug for DataObtainError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            DataObtainError::NonePeek => write!(f, "Called PeekResult::unwrap() on None value"),
            DataObtainError::ConversionError(a, b) => write!(f, "Conversion error from {} to {}", a, b),

            DataObtainError::StringParseError(s, ty) => write!(f, "Cannot parse String ({}) to type ({})", s, ty),

            DataObtainError::OtherAsObject(s) => write!(f, "Called obj_get on {} type", s),
            DataObtainError::NoSuchKey(key) => write!(f, "Object doesn't contain key \"{}\"", key),

            DataObtainError::OtherAsArray(s) => write!(f, "Called arr_get on {} type", s),
            DataObtainError::ArrayIndexOutOfBounds(idx, size) => write!(f, "Array index ({}) out of bounds [0 .. {})", idx, size),
        }
    }
}
impl std::fmt::Display for DataObtainError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        (self as &dyn std::fmt::Debug).fmt(f)
    }
}

/// Data field, can contain tree of other datas or array
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Data {
    // Null
    None,

    // Generics
    Bool(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    F32(f32),
    F64(f64),
    String(String), // String here may as well consider Generic

    // Object, map of other named datas
    Object(BTreeMap<String, Data>),

    // Array of other datas
    Array(Vec<Data>)
}

/// Result of converting data into primitive
#[derive(Debug, Clone, PartialEq)]
pub enum PeekResult<T, E> {
    Ok(T), // Good conversion
    Lossy(T), // Precision lost during conversion (Ex: u32 -> u8(potential precision), i32 -> u32(negatives), f32 -> u64(rounding from using `as`))
    Err(E), // Conversion cannot be done
}
impl <T, E: std::error::Error> PeekResult<T, E> {
    pub fn unwrap(self) -> T {
        use PeekResult::*;
        match self {
            Ok(v) | Lossy(v) => v,
            Err(error) => panic!("Called serializer::PeekResult::unwrap() on 'Error': {:?}", error),
        }
    }
    pub fn unwrap_or(self, or: T) -> T{
        use PeekResult::*;
        match self {
            Ok(v) | Lossy(v) => v,
            Err(_) => or,
        }
    }
}
impl <T, E: std::error::Error> Try for PeekResult<T, E> {
    type Ok = T;
    type Error = E;

    fn into_result(self) -> Result<T, E> {
        match self {
            PeekResult::Ok(v) | PeekResult::Lossy(v) => Ok(v),
            PeekResult::Err(e) => Err(e),
        }
    }

    fn from_error(v: E) -> Self { PeekResult::Err(v) }
    fn from_ok(v: T) -> Self { PeekResult::Ok(v) }
}

/// Peek is basically lossy into, converting inner fields for `Data`
/// Returns PeekResult, signifying precision of conversion
pub trait Peek<T, E> {
    /// Performs the conversion.
    fn peek(self) -> PeekResult<T, E>;
}

// #########################
// Peek primitives from Data
impl Peek<bool, DataObtainError> for Data {
    fn peek(self) -> PeekResult<bool, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "bool".to_string()))
            },
            Data::Bool(v) => Ok(v),
            Data::U8(v) => Lossy(v != 0),
            Data::U16(v) => Lossy(v != 0),
            Data::U32(v) => Lossy(v != 0),
            Data::U64(v) => Lossy(v != 0),
            Data::U128(v) => Lossy(v != 0),
            Data::I8(v) => Lossy(v != 0),
            Data::I16(v) => Lossy(v != 0),
            Data::I32(v) => Lossy(v != 0),
            Data::I64(v) => Lossy(v != 0),
            Data::I128(v) => Lossy(v != 0),
            Data::F32(v) => Lossy(v != 0.0),
            Data::F64(v) => Lossy(v != 0.0),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "bool".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "bool".to_string())),
        }
    }
}
impl Peek<u8, DataObtainError> for Data {
    fn peek(self) -> PeekResult<u8, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "u8".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1 } else { 0 }),
            Data::U8(v) => Ok(v),
            Data::U16(v) => Lossy(v as u8),
            Data::U32(v) => Lossy(v as u8),
            Data::U64(v) => Lossy(v as u8),
            Data::U128(v) => Lossy(v as u8),
            Data::I8(v) => Lossy(v as u8),
            Data::I16(v) => Lossy(v as u8),
            Data::I32(v) => Lossy(v as u8),
            Data::I64(v) => Lossy(v as u8),
            Data::I128(v) => Lossy(v as u8),
            Data::F32(v) => Lossy(v as u8),
            Data::F64(v) => Lossy(v as u8),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "u8".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "u8".to_string())),
        }
    }
}
impl Peek<u16, DataObtainError> for Data {
    fn peek(self) -> PeekResult<u16, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "u16".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1 } else { 0 }),
            Data::U8(v) => Ok(v as u16),
            Data::U16(v) => Ok(v),
            Data::U32(v) => Lossy(v as u16),
            Data::U64(v) => Lossy(v as u16),
            Data::U128(v) => Lossy(v as u16),
            Data::I8(v) => Lossy(v as u16),
            Data::I16(v) => Lossy(v as u16),
            Data::I32(v) => Lossy(v as u16),
            Data::I64(v) => Lossy(v as u16),
            Data::I128(v) => Lossy(v as u16),
            Data::F32(v) => Lossy(v as u16),
            Data::F64(v) => Lossy(v as u16),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "u16".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "u16".to_string())),
        }
    }
}
impl Peek<u32, DataObtainError> for Data {
    fn peek(self) -> PeekResult<u32, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "u32".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1 } else { 0 }),
            Data::U8(v) => Ok(v as u32),
            Data::U16(v) => Ok(v as u32),
            Data::U32(v) => Ok(v),
            Data::U64(v) => Lossy(v as u32),
            Data::U128(v) => Lossy(v as u32),
            Data::I8(v) => Lossy(v as u32),
            Data::I16(v) => Lossy(v as u32),
            Data::I32(v) => Lossy(v as u32),
            Data::I64(v) => Lossy(v as u32),
            Data::I128(v) => Lossy(v as u32),
            Data::F32(v) => Lossy(v as u32),
            Data::F64(v) => Lossy(v as u32),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "u32".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "u32".to_string())),
        }
    }
}
impl Peek<u64, DataObtainError> for Data {
    fn peek(self) -> PeekResult<u64, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "u64".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1 } else { 0 }),
            Data::U8(v) => Ok(v as u64),
            Data::U16(v) => Ok(v as u64),
            Data::U32(v) => Ok(v as u64),
            Data::U64(v) => Ok(v),
            Data::U128(v) => Lossy(v as u64),
            Data::I8(v) => Lossy(v as u64),
            Data::I16(v) => Lossy(v as u64),
            Data::I32(v) => Lossy(v as u64),
            Data::I64(v) => Lossy(v as u64),
            Data::I128(v) => Lossy(v as u64),
            Data::F32(v) => Lossy(v as u64),
            Data::F64(v) => Lossy(v as u64),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "u64".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "u64".to_string())),
        }
    }
}
impl Peek<u128, DataObtainError> for Data {
    fn peek(self) -> PeekResult<u128, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "u128".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1 } else { 0 }),
            Data::U8(v) => Ok(v as u128),
            Data::U16(v) => Ok(v as u128),
            Data::U32(v) => Ok(v as u128),
            Data::U64(v) => Ok(v as u128),
            Data::U128(v) => Ok(v),
            Data::I8(v) => Lossy(v as u128),
            Data::I16(v) => Lossy(v as u128),
            Data::I32(v) => Lossy(v as u128),
            Data::I64(v) => Lossy(v as u128),
            Data::I128(v) => Lossy(v as u128),
            Data::F32(v) => Lossy(v as u128),
            Data::F64(v) => Lossy(v as u128),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "u128".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "u128".to_string())),
        }
    }
}
impl Peek<i8, DataObtainError> for Data {
    fn peek(self) -> PeekResult<i8, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "i8".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1 } else { 0 }),
            Data::U8(v) => Lossy(v as i8),
            Data::U16(v) => Lossy(v as i8),
            Data::U32(v) => Lossy(v as i8),
            Data::U64(v) => Lossy(v as i8),
            Data::U128(v) => Lossy(v as i8),
            Data::I8(v) => Ok(v),
            Data::I16(v) => Lossy(v as i8),
            Data::I32(v) => Lossy(v as i8),
            Data::I64(v) => Lossy(v as i8),
            Data::I128(v) => Lossy(v as i8),
            Data::F32(v) => Lossy(v as i8),
            Data::F64(v) => Lossy(v as i8),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "i8".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "i8".to_string())),
        }
    }
}
impl Peek<i16, DataObtainError> for Data {
    fn peek(self) -> PeekResult<i16, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "i16".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1 } else { 0 }),
            Data::U8(v) => Ok(v as i16),
            Data::U16(v) => Lossy(v as i16),
            Data::U32(v) => Lossy(v as i16),
            Data::U64(v) => Lossy(v as i16),
            Data::U128(v) => Lossy(v as i16),
            Data::I8(v) => Ok(v as i16),
            Data::I16(v) => Ok(v),
            Data::I32(v) => Lossy(v as i16),
            Data::I64(v) => Lossy(v as i16),
            Data::I128(v) => Lossy(v as i16),
            Data::F32(v) => Lossy(v as i16),
            Data::F64(v) => Lossy(v as i16),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "i16".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "i16".to_string())),
        }
    }
}
impl Peek<i32, DataObtainError> for Data {
    fn peek(self) -> PeekResult<i32, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "i32".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1 } else { 0 }),
            Data::U8(v) => Ok(v as i32),
            Data::U16(v) => Ok(v as i32),
            Data::U32(v) => Lossy(v as i32),
            Data::U64(v) => Lossy(v as i32),
            Data::U128(v) => Lossy(v as i32),
            Data::I8(v) => Ok(v as i32),
            Data::I16(v) => Ok(v as i32),
            Data::I32(v) => Ok(v),
            Data::I64(v) => Lossy(v as i32),
            Data::I128(v) => Lossy(v as i32),
            Data::F32(v) => Lossy(v as i32),
            Data::F64(v) => Lossy(v as i32),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "i32".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "i32".to_string())),
        }
    }
}
impl Peek<i64, DataObtainError> for Data {
    fn peek(self) -> PeekResult<i64, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "i64".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1 } else { 0 }),
            Data::U8(v) => Ok(v as i64),
            Data::U16(v) => Ok(v as i64),
            Data::U32(v) => Ok(v as i64),
            Data::U64(v) => Lossy(v as i64),
            Data::U128(v) => Lossy(v as i64),
            Data::I8(v) => Ok(v as i64),
            Data::I16(v) => Ok(v as i64),
            Data::I32(v) => Ok(v as i64),
            Data::I64(v) => Ok(v),
            Data::I128(v) => Lossy(v as i64),
            Data::F32(v) => Lossy(v as i64),
            Data::F64(v) => Lossy(v as i64),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "i64".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "i64".to_string())),
        }
    }
}
impl Peek<i128, DataObtainError> for Data {
    fn peek(self) -> PeekResult<i128, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "i128".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1 } else { 0 }),
            Data::U8(v) => Ok(v as i128),
            Data::U16(v) => Ok(v as i128),
            Data::U32(v) => Ok(v as i128),
            Data::U64(v) => Ok(v as i128),
            Data::U128(v) => Lossy(v as i128),
            Data::I8(v) => Ok(v as i128),
            Data::I16(v) => Ok(v as i128),
            Data::I32(v) => Ok(v as i128),
            Data::I64(v) => Ok(v as i128),
            Data::I128(v) => Ok(v),
            Data::F32(v) => Lossy(v as i128),
            Data::F64(v) => Lossy(v as i128),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "i128".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "i128".to_string())),
        }
    }
}
impl Peek<f32, DataObtainError> for Data {
    fn peek(self) -> PeekResult<f32, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "f32".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1.0 } else { 0.0 }),
            Data::U8(v) => Ok(v as f32),
            Data::U16(v) => Ok(v as f32),
            Data::U32(v) => Ok(v as f32),
            Data::U64(v) => Ok(v as f32),
            Data::U128(v) => Ok(v as f32),
            Data::I8(v) => Ok(v as f32),
            Data::I16(v) => Ok(v as f32),
            Data::I32(v) => Ok(v as f32),
            Data::I64(v) => Ok(v as f32),
            Data::I128(v) => Ok(v as f32),
            Data::F32(v) => Ok(v),
            Data::F64(v) => Lossy(v as f32),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "f32".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "f32".to_string())),
        }
    }
}
impl Peek<f64, DataObtainError> for Data {
    fn peek(self) -> PeekResult<f64, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => match s.parse() {
                std::result::Result::Ok(v) => Ok(v),
                _ => Err(DataObtainError::StringParseError(s, "f64".to_string()))
            },
            Data::Bool(v) => Ok(if v { 1.0 } else { 0.0 }),
            Data::U8(v) => Ok(v as f64),
            Data::U16(v) => Ok(v as f64),
            Data::U32(v) => Ok(v as f64),
            Data::U64(v) => Ok(v as f64),
            Data::U128(v) => Ok(v as f64),
            Data::I8(v) => Ok(v as f64),
            Data::I16(v) => Ok(v as f64),
            Data::I32(v) => Ok(v as f64),
            Data::I64(v) => Ok(v as f64),
            Data::I128(v) => Ok(v as f64),
            Data::F32(v) => Ok(v as f64),
            Data::F64(v) => Ok(v),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "f64".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "f64".to_string())),
        }
    }
}
impl Peek<String, DataObtainError> for Data {
    fn peek(self) -> PeekResult<String, DataObtainError> {
        use PeekResult::*;
        match self {
            Data::String(s) => Ok(s),
            Data::Bool(v) => Ok(v.to_string()),
            Data::U8(v) => Ok(v.to_string()),
            Data::U16(v) => Ok(v.to_string()),
            Data::U32(v) => Ok(v.to_string()),
            Data::U64(v) => Ok(v.to_string()),
            Data::U128(v) => Ok(v.to_string()),
            Data::I8(v) => Ok(v.to_string()),
            Data::I16(v) => Ok(v.to_string()),
            Data::I32(v) => Ok(v.to_string()),
            Data::I64(v) => Ok(v.to_string()),
            Data::I128(v) => Ok(v.to_string()),
            Data::F32(v) => Ok(v.to_string()),
            Data::F64(v) => Ok(v.to_string()),
            Data::None => Err(DataObtainError::NonePeek),
            Data::Object(_) => Err(DataObtainError::ConversionError("Object".to_string(), "f64".to_string())),
            Data::Array(_) => Err(DataObtainError::ConversionError("Array".to_string(), "f64".to_string())),
        }
    }
}

// #########################
// Convert primitives into Data
impl From<bool> for Data    { fn from(v: bool)  -> Self { Data::Bool(v) } }
impl From<u8> for Data      { fn from(v: u8)    -> Self { Data::U8(v) } }
impl From<u16> for Data     { fn from(v: u16)   -> Self { Data::U16(v) } }
impl From<u32> for Data     { fn from(v: u32)   -> Self { Data::U32(v) } }
impl From<u64> for Data     { fn from(v: u64)   -> Self { Data::U64(v) } }
impl From<u128> for Data    { fn from(v: u128)  -> Self { Data::U128(v) } }
impl From<i8> for Data      { fn from(v: i8)    -> Self { Data::I8(v) } }
impl From<i16> for Data     { fn from(v: i16)   -> Self { Data::I16(v) } }
impl From<i32> for Data     { fn from(v: i32)   -> Self { Data::I32(v) } }
impl From<i64> for Data     { fn from(v: i64)   -> Self { Data::I64(v) } }
impl From<i128> for Data    { fn from(v: i128)  -> Self { Data::I128(v) } }
impl From<f32> for Data     { fn from(v: f32)   -> Self { Data::F32(v) } }
impl From<f64> for Data     { fn from(v: f64)   -> Self { Data::F64(v) } }
impl From<&str> for Data    { fn from(v: &str)  -> Self { Data::String(v.into()) } }
// Array
impl From<Vec<Data>> for Data { fn from(v: Vec<Data>)  -> Self { Data::Array(v) } }

// #########################
// Work with Object, Array, and None types for Data
/// Do all data obtaining by hand to make sure it is all ok and autocomplete works, so no macros
impl Data {

    // None works
    #[inline] pub fn has_data(&self) -> bool {
        match self {
            Data::None => false,
            _ => true,
        }
    }
    #[inline] pub fn is_none(&self) -> bool { !self.has_data() }

    // Object Works
    pub fn obj_get(&self, name: &str) -> PeekResult<Data, DataObtainError> {
        match self {
            Data::Object(tree) => {
                let s = name.to_string();
                match tree.get(&s) {
                    Some(v) => PeekResult::Ok(v.clone()),
                    None => PeekResult::Err(DataObtainError::NoSuchKey(s)),
                }
            },
            Data::Array(_) => PeekResult::Err(DataObtainError::OtherAsObject("Array".to_string())),
            _ => PeekResult::Err(DataObtainError::OtherAsObject("Primitive".to_string())),
        }
    }

    // Array Works
    pub fn arr_get(&self, idx: usize) -> PeekResult<Data, DataObtainError> {
        match self {
            Data::Array(v) => {
                match v.get(idx) {
                    Some(v) => PeekResult::Ok(v.clone()),
                    None => PeekResult::Err(DataObtainError::ArrayIndexOutOfBounds(idx, v.len()))
                }
            },
            Data::Object(_) => PeekResult::Err(DataObtainError::OtherAsObject("Object".to_string())),
            _ => PeekResult::Err(DataObtainError::OtherAsObject("Primitive".to_string())),
        }
    }
    pub fn arr_len(&self) -> PeekResult<usize, DataObtainError> {
        match self {
            Data::Array(v) => PeekResult::Ok(v.len()),
            Data::Object(_) => PeekResult::Err(DataObtainError::OtherAsObject("Object".to_string())),
            _ => PeekResult::Err(DataObtainError::OtherAsObject("Primitive".to_string())),
        }
    }

}

/// Macro creates Data::Object
#[allow(non_snake_case)]
macro_rules! DataObject {
    ( $( $key:expr => $val:expr ),* ) => {
        let mut tmp_map = std::collections::BTreeMap::new();
        $(
            tmp_map.insert(stringify!($key).to_string(), $val.into());
        )*
        Data::Object(tmp_map)
    }
}


/// Early Persistent serialization errors enum
pub enum PersistentError {
    UnableToDeserialize
}
impl std::error::Error for PersistentError {}
impl std::fmt::Debug for PersistentError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            PersistentError::UnableToDeserialize => write!(f, "Unable to deserialize"),
        }
    }
}
impl std::fmt::Display for PersistentError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        (self as &dyn std::fmt::Debug).fmt(f)
    }
}



/// Trait describes persistent value
/// Persistent uses Data to restore its state and returns data for serialization
pub trait Persistent {
    fn read(&mut self, val: Data) -> Result<(), PersistentError>;
    fn write(&self) -> Data;
}

mod test {
    use crate::{Peek, PeekResult, Persistent};
    use std::collections::BTreeMap;

    //noinspection RsApproxConstant
    #[test]
    fn test() {
        use crate::Data;

        // Simple convert
        assert_eq!(PeekResult::Ok(true), Data::String("true".to_string()).peek());
        assert_eq!(PeekResult::Ok(4u8), Data::String("4".to_string()).peek());
        assert_eq!(PeekResult::Ok(3.1415f32), Data::String("3.1415".to_string()).peek());
        assert_eq!(PeekResult::Ok(3.1415f64), Data::String("3.1415".to_string()).peek());

        // Data loss
        assert_eq!(PeekResult::Lossy(196u8), Data::I8(-60i8).peek()); // 255 (max u8) + 1 - 60

        // Serialization
        {
            let data_arr = Data::Array(vec![
                Data::U8(12),
                Data::I64(-12345),
                Data::F32(3.1415),
                Data::String("123".to_string()),
                Data::Object({
                    let mut map = BTreeMap::new();
                    map.insert("Key1".to_string(), Data::I32(12345));
                    map.insert("OtherKey".to_string(), Data::Array(vec![
                        Data::String("String1".to_string()),
                        Data::String("String2".to_string()),
                    ]));
                    map
                })
            ]);

            let serialized = serde_json::to_string(&data_arr).unwrap();
            let deserialized: Data = serde_json::from_str(serialized.as_str()).unwrap();
            let val: u16 = deserialized.arr_get(0).unwrap().peek().unwrap();
            assert_eq!(val, 12);
        }

        // Test Persistent Trait with version control
        {
            use crate::{ Persistent, PersistentError, Data };


            // Ver 1
            // Base version contains x: f32, y: f32
            #[derive(Default, Debug, Clone)]
            struct PointVer1 {
                x: f32,
                y: f32
            }
            impl Persistent for PointVer1 {
                fn read(&mut self, val: Data) -> Result<(), PersistentError> {
                    // Not required for this example
                    Ok(())
                }
                fn write(&self) -> Data {
                    DataObject! {
                        ver => 1u16,
                        x => self.x,
                        y => self.y
                    }
                }
            }

            // Update to ver: 2 added z: f32
            #[derive(Default, Debug, Clone, PartialEq)]
            struct PointVer2 {
                x: f32,
                y: f32,
                z: f32
            }
            impl Persistent for PointVer2 {
                fn read(&mut self, val: Data) -> Result<(), PersistentError> {
                    let ver: u16 = val.obj_get("ver").unwrap().peek().unwrap_or(0);
                    // or get data from object and assign it if data exist, whatever suits the case

                    // Matching ver approach
//                    match ver {
//                        1 => {
//                            self.x = val.obj_get("x").unwrap().peek().unwrap();
//                            self.y = val.obj_get("y").unwrap().peek().unwrap();
//                        },
//                        2 => {
//                            self.x = val.obj_get("x").unwrap().peek().unwrap();
//                            self.y = val.obj_get("y").unwrap().peek().unwrap();
//                            self.z = val.obj_get("z").unwrap().peek().unwrap();
//                        }
//                        _ => (), // No other versions exist, do nothing or panic or whatever
//                    }

                    // Appending ver patches approach
                    // if ver is known
                    if ver > 0 && ver <= 2 {
                        // Exists sense first version
                        self.x = val.obj_get("x").unwrap().peek().unwrap();
                        self.y = val.obj_get("y").unwrap().peek().unwrap();
                        // Only appeared in ver 2
                        if ver >= 2 {
                            self.z = val.obj_get("z").unwrap().peek().unwrap();
                        }
                        Ok(())
                    } else {
                        Err(PersistentError::UnableToDeserialize)
                    }
                }
                fn write(&self) -> Data {
                    DataObject! {
                        ver => 2,
                        x => self.x,
                        y => self.y,
                        z => self.z
                    }
                }
            }

            let p1 = PointVer1 {
                x: 1.0,
                y: 3.45
            };

            // Serialize into String
            let p1_ser = serde_json::to_string(&p1.write()).unwrap();

            // Get Data from String
            let p2_de: Data = serde_json::from_str(p1_ser.as_str()).unwrap();

            // Create new object but with different version and call read on it
            let mut p2 = PointVer2::default();
            p2.read(p2_de).unwrap();

            assert_eq!(PointVer2 {
                x: p1.x,
                y: p1.y,
                z: 0.0,
            }, p2);

            p2.z = 333.0;

            // try from ver 2 => ver 2
            let p2_ser = serde_json::to_string(&p2.write()).unwrap();

            let mut p3_de: Data = serde_json::from_str(p2_ser.as_str()).unwrap();
            let mut p3 = PointVer2::default();
            p3.read(p3_de).unwrap();
            assert_eq!(PointVer2 {
                x: p1.x,
                y: p1.y,
                z: 333.0,
            }, p2);

        }

    }
}


