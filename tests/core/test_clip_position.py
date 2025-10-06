#!/usr/bin/env python3
"""
Simple test script for the clip_new_position method
"""

import numpy as np
from contgrid.core.grid import WallCollisionChecker

def test_clip_new_position():
    """Test the clip_new_position method with various scenarios"""
    
    # Create a simple 3x3 grid with walls around the border
    layout = [
        "###",
        "#0#", 
        "###"
    ]
    
    L = 0.1  # Cell size
    checker = WallCollisionChecker(layout, L, verbose=True)
    
    R = 0.02  # Robot radius
    allowed_overlap = 0.01  # Small allowed overlap
    
    print("Testing analytical clip_new_position method...")
    print(f"Grid layout: {layout}")
    print(f"Cell size: {L}, Robot radius: {R}, Allowed overlap: {allowed_overlap}")
    print()
    
    # Test case 1: Valid movement within free space
    curr_pos = (0.0, 0.0)  # Center of middle cell
    new_pos = (0.02, 0.02)  # Small movement within free space
    clipped = checker.clip_new_position(R, allowed_overlap, curr_pos, new_pos)
    print("Test 1 - Valid movement:")
    print(f"  Current: {curr_pos}, New: {new_pos}")
    print(f"  Clipped: {clipped}")
    print(f"  Should be same as new_pos: {np.allclose(clipped, new_pos)}")
    print()
    
    # Test case 2: Movement that would collide with wall
    curr_pos = (0.0, 0.0)  # Center of middle cell
    new_pos = (0.08, 0.0)  # Try to move close to right wall
    clipped = checker.clip_new_position(R, allowed_overlap, curr_pos, new_pos)
    print("Test 2 - Movement towards wall:")
    print(f"  Current: {curr_pos}, New: {new_pos}")
    print(f"  Clipped: {clipped}")
    print(f"  Clipped should be between current and new: {curr_pos[0] <= clipped[0] <= new_pos[0]}")
    print(f"  Clipped position valid: {checker.is_position_valid(R, allowed_overlap, clipped)}")
    print()
    
    # Test case 3: No movement
    curr_pos = (0.0, 0.0)
    new_pos = (0.0, 0.0)
    clipped = checker.clip_new_position(R, allowed_overlap, curr_pos, new_pos)
    print("Test 3 - No movement:")
    print(f"  Current: {curr_pos}, New: {new_pos}")
    print(f"  Clipped: {clipped}")
    print(f"  Should be same as current: {np.allclose(clipped, curr_pos)}")
    print()
    
    # Test case 4: Movement in negative direction
    curr_pos = (0.0, 0.0)
    new_pos = (-0.08, 0.0)  # Try to move close to left wall
    clipped = checker.clip_new_position(R, allowed_overlap, curr_pos, new_pos)
    print("Test 4 - Movement towards left wall:")
    print(f"  Current: {curr_pos}, New: {new_pos}")
    print(f"  Clipped: {clipped}")
    print(f"  Clipped should be between new and current: {new_pos[0] <= clipped[0] <= curr_pos[0]}")
    print(f"  Clipped position valid: {checker.is_position_valid(R, allowed_overlap, clipped)}")
    print()
    
    # Test case 5: Diagonal movement towards corner
    curr_pos = (0.0, 0.0)
    new_pos = (0.08, 0.08)  # Try to move towards top-right
    clipped = checker.clip_new_position(R, allowed_overlap, curr_pos, new_pos)
    print("Test 5 - Diagonal movement towards corner:")
    print(f"  Current: {curr_pos}, New: {new_pos}")
    print(f"  Clipped: {clipped}")
    print(f"  Clipped position valid: {checker.is_position_valid(R, allowed_overlap, clipped)}")
    print()
    
    # Test case 6: Test analytical approach efficiency
    import time
    print("Performance test:")
    start_time = time.time()
    for i in range(1000):
        test_pos = (0.0, 0.0)
        target_pos = (0.05 + i * 0.0001, 0.05 + i * 0.0001)
        clipped = checker.clip_new_position(R, allowed_overlap, test_pos, target_pos)
    end_time = time.time()
    print(f"  1000 clip operations took: {(end_time - start_time) * 1000:.2f} ms")
    print(f"  Average per operation: {(end_time - start_time) * 1000000 / 1000:.2f} μs")

if __name__ == "__main__":
    test_clip_new_position()