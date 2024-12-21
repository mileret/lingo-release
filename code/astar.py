from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import numpy as np
import cv2

def get_path(start, end, synhsi_dataset):
    voxel_size = np.divide(synhsi_dataset.scene_grid_np[3: 6] - synhsi_dataset.scene_grid_np[:3],
                           synhsi_dataset.scene_grid_np[6:])[[0, 2]]
    grid_start = np.divide((start - synhsi_dataset.scene_grid_np[[0, 2]]), voxel_size).astype(int)
    grid_end = np.divide((end - synhsi_dataset.scene_grid_np[[0, 2]]), voxel_size).astype(int)

    occ = synhsi_dataset.scene_occ[0].detach().cpu().numpy()

    occ_grid = np.sum(occ[:, 10: 40, :], axis=1) # 400x600
    occ_grid = (occ_grid - occ_grid.min()) / (occ_grid.max() - occ_grid.min()) * 255

    occ_grid = cv2.dilate(occ_grid, np.ones((15, 15), np.uint8), iterations=1)

    occ_grid = (occ_grid - occ_grid.min()) / (occ_grid.max() - occ_grid.min()) * 255 + 1
    occ_grid[occ_grid > 1] = 255

    img = (occ_grid - occ_grid.min()) / (occ_grid.max() - occ_grid.min()) * 255
    cv2.imwrite('gray0.jpg', img.T)

    occ_grid_astar = Grid(matrix=occ_grid.T)

    start = occ_grid_astar.node(grid_start[0], grid_start[1])
    end = occ_grid_astar.node(grid_end[0], grid_end[1])
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, occ_grid_astar)

    path = np.array([[p.x, p.y] for p in path])

    midpoints = path * voxel_size + synhsi_dataset.scene_grid_np[[0, 2]]
    midpoints = midpoints[list(range(0, len(midpoints), 1))]

    # draw the trajectory on the new rgb image
    img_new = np.zeros((img.shape[0], img.shape[1], 3))
    img_new[:, :, 0] = img
    for i in range(len(path)):
        img_new[int(path[i, 0]), int(path[i, 1]), 1] = 255

    cv2.imwrite(f'gray1.jpg', img_new.transpose(1, 0, 2))

    return midpoints