import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Import all functions from Assignment5
from Assignment5 import (
    chain_rule, ReLU, chain_rule_a, chain_rule_b,
    backprop_a, backprop_b, backprop_c,
    newtonMethod, sgd, constructParaboloid
)

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_html_report():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Assignment 5 Report - Computer Vision</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                max-width: 1200px;
                margin: 40px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                margin-top: 30px;
                background-color: #ecf0f1;
                padding: 10px;
                border-left: 5px solid #3498db;
            }
            h3 {
                color: #555;
                margin-top: 20px;
            }
            .task {
                background-color: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .output {
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 4px solid #28a745;
                font-family: 'Courier New', monospace;
                margin: 10px 0;
            }
            .image-container {
                text-align: center;
                margin: 20px 0;
            }
            img {
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #3498db;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <h1>EEP 596 Computer Vision - Assignment 5 Report</h1>
        <p><strong>PyTorch and Backpropagation</strong></p>
    """
    
    # Task 1: Chain Rule
    html_content += """
        <div class="task">
            <h2>Task 1: Chain Rule</h2>
            <p>For function f(x,y,z) = xy+z, compute df/dz, df/dq, df/dx, and df/dy where q=xy.</p>
            <p>Input values: x=-2, y=5, z=-4</p>
    """
    
    df_dz, df_dq, df_dx, df_dy = chain_rule()
    html_content += f"""
            <div class="output">
                <table>
                    <tr><th>Derivative</th><th>Value</th></tr>
                    <tr><td>df/dz</td><td>{df_dz}</td></tr>
                    <tr><td>df/dq</td><td>{df_dq}</td></tr>
                    <tr><td>df/dx</td><td>{df_dx}</td></tr>
                    <tr><td>df/dy</td><td>{df_dy}</td></tr>
                </table>
            </div>
        </div>
    """
    
    # Task 2: ReLU
    html_content += """
        <div class="task">
            <h2>Task 2: ReLU Backpropagation</h2>
            <p>For input data=[-1,-2] and weights=[2,-3,-3], compute dx and dw after backpropagation.</p>
    """
    
    dx, dw = ReLU()
    html_content += f"""
            <div class="output">
                <p><strong>Gradient w.r.t. inputs (dx):</strong> {dx}</p>
                <p><strong>Gradient w.r.t. weights (dw):</strong> {dw}</p>
            </div>
        </div>
    """
    
    # Task 3a: Chain Rule with PyTorch
    html_content += """
        <div class="task">
            <h2>Task 3: Chain Rule with PyTorch</h2>
            <h3>Part (a): Forward Pass Values</h3>
            <p>Calculate a, b, c values for f(w,x) = 1/(1 + exp(-(w0x0+w1x1+w2)))</p>
    """
    
    a_val, b_val, c_val = chain_rule_a()
    html_content += f"""
            <div class="output">
                <table>
                    <tr><th>Variable</th><th>Value</th></tr>
                    <tr><td>a (exp(-s3))</td><td>{a_val}</td></tr>
                    <tr><td>b (a + 1)</td><td>{b_val}</td></tr>
                    <tr><td>c (1/b)</td><td>{c_val}</td></tr>
                </table>
            </div>
    """
    
    # Task 3b: Backward Pass
    html_content += """
            <h3>Part (b): Backward Pass Gradients</h3>
    """
    
    gradients = chain_rule_b()
    grad_names = ['∂c/∂w0', '∂c/∂x0', '∂c/∂w1', '∂c/∂x1', '∂c/∂w2']
    html_content += """
            <div class="output">
                <table>
                    <tr><th>Gradient</th><th>Value</th></tr>
    """
    for name, grad in zip(grad_names, gradients):
        html_content += f"<tr><td>{name}</td><td>{grad.item()}</td></tr>"
    html_content += """
                </table>
            </div>
        </div>
    """
    
    # Task 4: Backpropagation
    html_content += """
        <div class="task">
            <h2>Task 4: Backpropagation with tanh</h2>
            <p>For f(w,x) = tanh(w0x0+w1x1+w2) with w=[5,2], x=[-1,4], w2=-2, and MSE loss with ground truth=1</p>
    """
    
    # Part a
    y_hat = backprop_a()
    html_content += f"""
            <h3>Part (a): Forward Pass</h3>
            <div class="output">
                <p><strong>Output y_hat = f(w,x):</strong> {y_hat.item():.6f}</p>
            </div>
    """
    
    # Part b
    gw0, gw1, gw2 = backprop_b()
    html_content += f"""
            <h3>Part (b): Gradients</h3>
            <div class="output">
                <table>
                    <tr><th>Weight</th><th>Gradient</th></tr>
                    <tr><td>∂L/∂w0</td><td>{gw0.item():.6f}</td></tr>
                    <tr><td>∂L/∂w1</td><td>{gw1.item():.6f}</td></tr>
                    <tr><td>∂L/∂w2</td><td>{gw2.item():.6f}</td></tr>
                </table>
            </div>
    """
    
    # Part c
    w0_new, w1_new, w2_new = backprop_c()
    html_content += f"""
            <h3>Part (c): Updated Weights (learning rate = 0.1)</h3>
            <div class="output">
                <table>
                    <tr><th>Weight</th><th>Original</th><th>Updated</th></tr>
                    <tr><td>w0</td><td>5.0</td><td>{w0_new.item():.6f}</td></tr>
                    <tr><td>w1</td><td>2.0</td><td>{w1_new.item():.6f}</td></tr>
                    <tr><td>w2</td><td>-2.0</td><td>{w2_new.item():.6f}</td></tr>
                </table>
            </div>
        </div>
    """
    
    # Task 5: Optimization
    html_content += """
        <div class="task">
            <h2>Task 5: Optimization Methods</h2>
            <p>Finding the minimum of a 2D paraboloid using different optimization methods.</p>
    """
    
    # Generate paraboloid visualization
    paraboloid = constructParaboloid()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(paraboloid, cmap='viridis')
    ax.set_title('Paraboloid Function')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    paraboloid_img = fig_to_base64(fig)
    
    html_content += f"""
            <div class="image-container">
                <h3>Paraboloid Visualization</h3>
                <img src="data:image/png;base64,{paraboloid_img}" alt="Paraboloid">
            </div>
    """
    
    # Part a: Newton's Method
    html_content += """
            <h3>Part (a): Newton's Method</h3>
            <p>Starting from position (200, 50)</p>
    """
    
    print("\n=== Newton's Method ===")
    newton_x, newton_y, newton_iters = newtonMethod(200, 50)
    
    # Visualize Newton's path
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(paraboloid, cmap='viridis', alpha=0.7)
    ax.plot(200, 50, 'ro', markersize=10, label='Start (200, 50)')
    ax.plot(newton_x, newton_y, 'g*', markersize=15, label=f'Newton End ({newton_x}, {newton_y})')
    ax.set_title("Newton's Method Convergence")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    newton_img = fig_to_base64(fig)
    
    html_content += f"""
            <div class="output">
                <p><strong>Converged to:</strong> ({newton_x}, {newton_y})</p>
                <p><strong>Iterations to converge:</strong> {newton_iters}</p>
            </div>
            <div class="image-container">
                <img src="data:image/png;base64,{newton_img}" alt="Newton's Method">
            </div>
    """
    
    # Part b: SGD
    html_content += """
            <h3>Part (b): Stochastic Gradient Descent (SGD)</h3>
            <p>Starting from position (200, 50) with learning rate = 0.001</p>
    """
    
    print("\n=== SGD ===")
    sgd_x, sgd_y, sgd_iters = sgd(200, 50, lr=0.001)
    
    # Visualize SGD path
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(paraboloid, cmap='viridis', alpha=0.7)
    ax.plot(200, 50, 'ro', markersize=10, label='Start (200, 50)')
    ax.plot(sgd_x, sgd_y, 'b*', markersize=15, label=f'SGD End ({sgd_x}, {sgd_y})')
    ax.set_title('SGD Convergence (lr=0.001)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    sgd_img = fig_to_base64(fig)
    
    html_content += f"""
            <div class="output">
                <p><strong>Converged to:</strong> ({sgd_x}, {sgd_y})</p>
                <p><strong>Iterations to converge:</strong> {sgd_iters}</p>
            </div>
            <div class="image-container">
                <img src="data:image/png;base64,{sgd_img}" alt="SGD">
            </div>
    """
    
    # Learning rate analysis
    html_content += """
            <h3>Learning Rate Analysis</h3>
            <div class="output">
                <p><strong>1) Effect of learning rate:</strong></p>
                <ul>
                    <li><strong>Small learning rate (e.g., 0.00001):</strong> Slower convergence but more stable</li>
                    <li><strong>Medium learning rate (e.g., 0.001):</strong> Good balance between speed and stability</li>
                    <li><strong>Large learning rate (e.g., 1.0):</strong> May converge faster or cause divergence/oscillation</li>
                </ul>
                <p><strong>2) Divergence:</strong></p>
                <p>A learning rate that is too large (e.g., > 1.0) can cause divergence, where the optimization 
                   overshoots the minimum and the values oscillate or grow without bound. For this paraboloid,
                   learning rates above approximately 1.0-2.0 may cause divergence depending on the starting position.</p>
            </div>
    """
    
    # Test different learning rates
    print("\n=== Testing different learning rates ===")
    lr_tests = [0.00001, 0.001, 1.0]
    lr_results = []
    
    for lr in lr_tests:
        try:
            x, y, iters = sgd(200, 50, lr=lr)
            lr_results.append((lr, x, y, iters))
            print(f"LR={lr}: Converged to ({x}, {y}) in {iters} iterations")
        except Exception as e:
            lr_results.append((lr, None, None, None))
            print(f"LR={lr}: Failed/Diverged - {str(e)}")
    
    html_content += """
            <h3>Learning Rate Experiments</h3>
            <div class="output">
                <table>
                    <tr><th>Learning Rate</th><th>Final Position</th><th>Iterations to Converge</th></tr>
    """
    
    for lr, x, y, iters in lr_results:
        if x is not None:
            html_content += f"<tr><td>{lr}</td><td>({x}, {y})</td><td>{iters}</td></tr>"
        else:
            html_content += f"<tr><td>{lr}</td><td>Failed/Diverged</td><td>-</td></tr>"
    
    html_content += """
                </table>
            </div>
        </div>
    """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    return html_content

if __name__ == "__main__":
    print("Generating report...")
    
    # Create report directory if it doesn't exist
    report_dir = os.path.join(os.path.dirname(__file__), 'report')
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate and save report
    html_content = generate_html_report()
    
    report_path = os.path.join(report_dir, 'assignment5_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nReport generated successfully: {report_path}")
    print("You can open this file in a browser and print to PDF.")
