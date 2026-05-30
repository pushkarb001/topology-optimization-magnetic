import femm
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'shape_representation'))

from ngn import make_grid, make_centers, ngn_density, threshold
from rbf import rbf_density, normalize


def simulate(binary_matrix, gap=5.0, verbose=False):
    """
    binary_matrix : 2D numpy array, 1=iron, 0=air
    returns flux linkage through the coil in Wb
    """

    femm.openfemm(1)
    femm.newdocument(0)
    femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, 0, 30)

    femm.mi_getmaterial('Air')
    femm.mi_getmaterial('Pure Iron')
    femm.mi_addmaterial('Copper', 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0)
    femm.mi_addcircprop('Coil', 1, 1)

    aw = 10
    iw = 20
    il = 30
    ox = 6
    cw = 4

    tw  = ox + aw + iw + aw
    th  = aw + il + aw

    coil_left_x1  = ox - cw
    coil_left_x2  = ox
    coil_right_x1 = ox + aw
    coil_right_x2 = ox + aw + cw
    coil_y1 = aw + 2
    coil_y2 = th - aw - 2

    # the right side of the core is the design region
    # the binary map fills this with iron or air cells
    dr_x0 = ox + aw + iw
    dr_x1 = tw
    dr_y0 = aw
    dr_y1 = th - aw
    dr_w  = dr_x1 - dr_x0
    dr_h  = dr_y1 - dr_y0

    rows, cols = binary_matrix.shape
    cell_w = dr_w / cols
    cell_h = dr_h / rows

    # fixed C-core — left arm, top and bottom arms stay the same
    # only the right side changes based on the binary map
    femm.mi_drawline(tw,   0,    ox,   0   )
    femm.mi_drawline(ox,   0,    ox,   th  )
    femm.mi_drawline(ox,   th,   tw,   th  )
    femm.mi_drawline(tw,   th,   tw,   th-aw)
    femm.mi_drawline(tw,   aw,   tw,   0   )
    femm.mi_drawline(ox+aw, th-aw, ox+aw, aw)
    femm.mi_drawline(ox+aw, aw,   dr_x0, aw)
    femm.mi_drawline(ox+aw, th-aw, dr_x0, th-aw)

    # boundary of the design region
    femm.mi_drawline(dr_x0, dr_y0, dr_x0, dr_y1)
    femm.mi_drawline(dr_x0, dr_y1, dr_x1, dr_y1)
    femm.mi_drawline(dr_x1, dr_y1, dr_x1, dr_y0)
    femm.mi_drawline(dr_x1, dr_y0, dr_x0, dr_y0)

    # coil around left arm
    femm.mi_drawline(coil_left_x1, coil_y1, coil_left_x1, coil_y2)
    femm.mi_drawline(coil_left_x1, coil_y1, coil_left_x2, coil_y1)
    femm.mi_drawline(coil_left_x1, coil_y2, coil_left_x2, coil_y2)
    femm.mi_drawline(coil_right_x2, coil_y1, coil_right_x2, coil_y2)
    femm.mi_drawline(coil_right_x1, coil_y1, coil_right_x2, coil_y1)
    femm.mi_drawline(coil_right_x1, coil_y2, coil_right_x2, coil_y2)

    # each cell in the design region gets drawn and labeled
    for row in range(rows):
        for col in range(cols):
            x0 = dr_x0 + col * cell_w
            x1 = dr_x0 + (col + 1) * cell_w
            y0 = dr_y0 + row * cell_h
            y1 = dr_y0 + (row + 1) * cell_h
            femm.mi_drawrectangle(x0, y0, x1, y1)

    # iron label for the fixed C-core body
    femm.mi_addblocklabel((ox + tw)/2, aw/2)
    femm.mi_selectlabel((ox + tw)/2, aw/2)
    femm.mi_setblockprop('Pure Iron', 1, 0, '<None>', 0, 0, 0)
    femm.mi_clearselected()

    femm.mi_addblocklabel((coil_left_x1 + coil_left_x2)/2, th/2)
    femm.mi_selectlabel((coil_left_x1 + coil_left_x2)/2, th/2)
    femm.mi_setblockprop('Copper', 1, 0, 'Coil', 0, 0, 500)
    femm.mi_clearselected()

    femm.mi_addblocklabel((coil_right_x1 + coil_right_x2)/2, th/2)
    femm.mi_selectlabel((coil_right_x1 + coil_right_x2)/2, th/2)
    femm.mi_setblockprop('Copper', 1, 0, 'Coil', 0, 0, -500)
    femm.mi_clearselected()

    # assign iron or air to each cell based on the binary map
    for row in range(rows):
        for col in range(cols):
            cx = dr_x0 + (col + 0.5) * cell_w
            cy = dr_y0 + (row + 0.5) * cell_h
            material = 'Pure Iron' if binary_matrix[row, col] == 1 else 'Air'
            femm.mi_addblocklabel(cx, cy)
            femm.mi_selectlabel(cx, cy)
            femm.mi_setblockprop(material, 1, 0, '<None>', 0, 0, 0)
            femm.mi_clearselected()

    # air inside the open window of the C
    femm.mi_addblocklabel(coil_right_x2 + 2, th/2)
    femm.mi_selectlabel(coil_right_x2 + 2, th/2)
    femm.mi_setblockprop('Air', 1, 0, '<None>', 0, 0, 0)
    femm.mi_clearselected()

    femm.mi_addblocklabel(tw + 40, th/2)
    femm.mi_selectlabel(tw + 40, th/2)
    femm.mi_setblockprop('Air', 1, 0, '<None>', 0, 0, 0)
    femm.mi_clearselected()

    femm.mi_makeABC()
    femm.mi_saveas('ccore_opt.fem')
    femm.mi_analyze()
    femm.mi_loadsolution()

    if verbose:
        B_left = femm.mo_getb(ox + aw/2, th/2)
        B_gap  = femm.mo_getb(tw - aw/2, th/2)
        print(f'Left arm: By={B_left[1]:.4f} T')
        print(f'Gap:      By={B_gap[1]:.4f} T')

    circuit = femm.mo_getcircuitproperties('Coil')
    flux_linkage = abs(circuit[2])

    femm.mo_close()
    femm.mi_close()
    femm.closefemm()

    return flux_linkage


if __name__ == '__main__':
    binary_matrix = np.ones((15, 5))
    result = simulate(binary_matrix, verbose=True)
    print(f'Flux linkage (solid iron): {result:.6f} Wb')

    np.random.seed(42)
    binary_matrix = (np.random.rand(15, 5) > 0.5).astype(float)
    result = simulate(binary_matrix, verbose=True)
    print(f'Flux linkage (random): {result:.6f} Wb')

    binary_matrix = np.zeros((15, 5))
    result = simulate(binary_matrix, verbose=True)
    print(f'Flux linkage (all air): {result:.6f} Wb')