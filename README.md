<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h2>Project Team</h2>
    <table>
        <tr>
            <th>Name</th>
            <th>USN</th>
        </tr>
        <tr>
            <td>Anusha Venkatesh</td>
            <td>A20594323</td>
        </tr>
        <tr>
            <td>Gurukiran Shivashankar</td>
            <td>A20564280</td>
        </tr>
        <tr>
            <td>Rachana Vijay</td>
            <td>A20605843</td>
        </tr>
        <tr>
            <td>Shanika Kadidal Sundresh</td>
            <td>A20585446</td>
        </tr>
    </table>


<h1>How to Run the Project</h1>
<p>To set up and execute the project, follow these steps using a virtual environment:</p>

<h3>Set execution policy:</h3>
<pre><code>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass</code></pre>

<h3>Activate the virtual environment:</h3>
<pre><code> .\venv\Scripts\Activate</code></pre>

<h3>Generate the regression dataset:</h3>
<pre><code> python tests/generate_regression_data.py</code></pre>

<h3>Set the PYTHONPATH environment variable:</h3>
<pre><code> $env:PYTHONPATH="."</code></pre>

<h3>Run the test suite:</h3>
<pre><code> pytest tests/ -s</code></pre>

<hr>

<h2>1. What does the model you have implemented do, and when should it be used?</h2>
<p>This project implements a LASSO-regularized linear regression model using the Homotopy approach, which follows a greedy coordinate-descent-like strategy. LASSO (Least Absolute Shrinkage and Selection Operator) applies an L1 penalty to the regression cost function, promoting sparsity by driving some coefficients to exactly zero.</p>

<h3>This model is particularly useful in the following scenarios:</h3>
<ul>
    <li>When only a subset of features is expected to be important (feature selection).</li>
    <li>In cases of multicollinearity among features, where it helps reduce redundancy.</li>
    <li>When an interpretable model is desired, as it eliminates irrelevant or redundant features.</li>
</ul>

<hr>

<h2>2. How did you test your model to determine if it is working correctly?</h2>
<p>To ensure correctness, multiple test cases were implemented using PyTest:</p>
<ul>
    <li>✅ <b>test_small_dataset</b> – Verifies model behavior on a simple CSV dataset.</li>
    <li>✅ <b>test_collinear_data_sparse_solution</b> – Tests the model’s ability to handle highly collinear data and produce sparse solutions.</li>
    <li>✅ <b>test_generated_data_sparse_solution</b> – Uses an auto-generated dataset (via <code>generate_regression_data.py</code>) with known coefficients to validate both learning and sparsity.</li>
    <li>✅ <b>test_irrelevant_feature</b> – Evaluates whether the model suppresses irrelevant features by setting their coefficients near zero.</li>
</ul>

<hr>

<h2>3. What parameters have you exposed to users for tuning performance?</h2>
<p>The following parameters can be adjusted when initializing the <code>LassoHomotopyModel</code>:</p>

<table border="1">
    <tr>
        <th>Parameter</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><b>alpha</b></td>
        <td>Controls the strength of the L1 penalty (regularization).</td>
    </tr>
    <tr>
        <td><b>max_iter</b></td>
        <td>Specifies the maximum number of iterations in the homotopy optimization process.</td>
    </tr>
    <tr>
        <td><b>tol</b></td>
        <td>Sets the tolerance threshold for stopping, based on residual norm.</td>
    </tr>
</table>

<p>Users can configure these parameters as follows:</p>
<pre><code>model = LassoHomotopyModel(alpha=0.1, max_iter=1000, tol=1e-4)</code></pre>

<hr>

<h2>4. Are there specific inputs that your implementation has trouble with? Could improvements be made?</h2>
<p>The model may struggle in the following cases:</p>
<ul>
    <li>When all features are equally important and uncorrelated (sparse solutions are less useful here).</li>
    <li>If features have vastly different scales (partially mitigated through normalization).</li>
    <li>Handling very large datasets, where least squares calculations can be computationally expensive.</li>
</ul>

<h3>Potential improvements with more time could include:</h3>
<ul>
    <li>Implementing <b>warm-starting</b> to improve efficiency.</li>
    <li>Adaptive step-size selection for faster convergence.</li>
    <li>Using efficient matrix updates, such as Cholesky factorization, to enhance performance and stability.</li>
</ul>

</body>
</html>
