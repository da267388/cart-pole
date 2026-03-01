import numpy as np

def cartpole_linearized(m_cart=1.0, m_pole=0.1, length=0.5, gravity=9.81):
    """
    Returns the linearized A, B matrices of the cart-pole around upright position (theta=0).
    States: x = [x, x_dot, theta, theta_dot]
    Input:  u = horizontal force on cart
    """
    """
    x: cart position
    x_dot: cart velocity
    theta: pole angle from upright (rad, theta=0 is upright)
    theta_dot: pole angular velocity
    
    """
    # x_dot = Ax + Bu
    
    M = m_cart
    m = m_pole
    l = length
    g = gravity
    # samplified for linearization
    A = np.array([
        [0,     1,             0,                0], # position_dot = velocity + Bu
        [0,     0,     (m * g)/M,               0], # velocity_dot = (force / cart_mass) * pole_angle + Bu
        [0,     0,             0,                1], # pole_angle_dot = pole_angular_velocity + Bu
        [0,     0, g*(M+m)/(M*l),               0] # pole_angular_velocity_dot = g * (total_mass) / (cart_mass * length) pole_angle (torque equilibrium)
    ])

    B = np.array([
        [0], # position doesn't directly affect by force
        [1/M], # velocity_dot = Ax + force / cart_mass
        [0], # pole angle doesn't directly affect by force
        [1/(M*l)] # pole_angular_velocity_dot = Ax + force / (cart_mass * length)
    ])

    return A, B