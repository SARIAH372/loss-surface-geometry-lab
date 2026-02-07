# Loss Surface Geometry Lab

An interactive Streamlit application for exploring how calculus governs learning in machine learning models.

This app visualizes loss landscapes, gradients, curvature (via Hessian eigenvalues), and optimization dynamics in a two-parameter setting. Rather than focusing on applications, it treats optimization itself as the object of study.

## What this app shows

- Loss surface contours in parameter space  
- Gradient vector fields and gradient flow  
- Hessian eigenvalues (local curvature, saddles vs minima)  
- Optimization paths for:
  - Gradient Descent / Momentum
  - Newton’s Method (damped)
  - Trust-Region methods
  - Implicit gradient steps
  - Adam
- Stability behavior as step size varies
- Exportable optimization traces for further analysis

All computations update dynamically through user interaction.

## Why this exists

Modern machine learning learns by calculus.  
This project makes that explicit by visualizing how gradients, curvature, and numerical methods shape optimization behavior.

No domain assumptions, no application claims — only structure, geometry, and dynamics.

## Tech stack

- Python
- Streamlit
- NumPy
- Matplotlib

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py

