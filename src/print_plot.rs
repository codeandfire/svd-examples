use ndarray::prelude::*;
use plotters::coord::types::RangedCoordf32;
use plotters::prelude::*;
use std::fmt::Display;
use std::path::Path;

/// Print out a matrix as a neatly formatted table, with labels along the rows
/// and columns.
/// `width` and `precision` are formatting attributes.
pub fn print_matrix<T: Display>(
    matrix: &Array<f64, Ix2>,
    row_labels: &[T],
    col_labels: &[T],
    width: usize,
    precision: usize
) -> Result<(), &'static str> {
    if (row_labels.len(), col_labels.len()) != matrix.dim() {
        return Err("Dimensions of array do not match number of row and/or column labels.");
    }

    for _ in 0..width + 1 {
        print!(" ");
    }

    for label in col_labels.iter() {
        print!("{:>w$} ", label, w = width);
    }
    println!();

    for (i, label) in row_labels.iter().enumerate() {
        print!("{:<w$} ", label, w = width);

        for j in 0..col_labels.len() {
            print!("{:>w$.p$} ", matrix[[i, j]], w = width, p = precision);
        }
        println!();
    }

    Ok(())
}

/// Print out a set of vectors as a neatly formatted table, with labels along the rows.
/// `width` and `precision` are formatting attributes.
pub fn print_vectors<T: Display>(
    vectors: &Array<f64, Ix2>,
    labels: &[T],
    width: usize,
    precision: usize,
) -> Result<(), &'static str> {
    if vectors.len_of(Axis(0)) != labels.len() {
        return Err("Number of rows in array do not match number of labels.");
    }

    for (label, vec) in labels.iter().zip(vectors.genrows()) {
        println!("{:<w$} {:>w$.p$}", label, vec, w = width, p = precision);
    }

    Ok(())
}

/// Plot a set of vectors along with labels on a graph.
pub fn plot_vectors<T, P>(vectors: &Array<f32, Ix2>, labels: &[T], save_file: P) -> Result<(), &'static str>
where T: Display,
      P: AsRef<Path> + Display
{
    if vectors.len_of(Axis(0)) != labels.len() {
        return Err("Number of rows in array do not match number of labels.");
    }

    const FIGSIZE: u32 = 500;
    const PADDING: u32 = 50; // to accommodate for any text overflow from the labels.

    // find the maximum magnitude of all elements in the vectors and use that
    // to set up a square coordinate system.
    let max_value = vectors.iter().map(|v| v.abs()).reduce(f32::max).unwrap();
    let coord: Cartesian2d<RangedCoordf32, RangedCoordf32> = Cartesian2d::new(
        -max_value..max_value,
        -max_value..max_value,
        (
            PADDING as i32..(FIGSIZE - PADDING) as i32,
            PADDING as i32..(FIGSIZE - PADDING) as i32,
        ),
    );

    let root = SVGBackend::new(&save_file, (FIGSIZE, FIGSIZE))
        .into_drawing_area()
        .apply_coord_spec(coord);
    root.fill(&WHITE).expect("Failed to fill figure.");

    let plot_dot_and_label = |x: f32, y: f32, label: &T| {
        EmptyElement::at((x, y))
            + Circle::new((0, 0), 3, ShapeStyle::from(&BLACK).filled())
            + Text::new(
                format!("{}", label),
                (10, 0),
                ("sans-serif", 15.0).into_font(),
            )
    };

    for (vector, label) in vectors
        .genrows()
        .into_iter()
        .map(|row| row.to_vec())
        .zip(labels.iter())
    {
        let (x, y) = (vector[0], vector[1]);
        root.draw(&plot_dot_and_label(x, y, label))
            .expect("Failed to draw on figure.");
    }

    println!("Plot saved to {}", save_file);

    Ok(())
}
