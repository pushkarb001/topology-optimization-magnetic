import femm


def run_ccore_simulation(gap=5.0):

    femm.openfemm()
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
    tip = (il - gap) / 2

    # coil dimensions
    coil_left_x1  = ox - cw   # x=2
    coil_left_x2  = ox        # x=6  
    coil_right_x1 = ox + aw   # x=16 
    coil_right_x2 = ox + aw + cw  # x=20
    coil_y1 = aw + 2           # y=12
    coil_y2 = th - aw - 2      # y=38

    # C-core outline
    femm.mi_drawline(tw,    0,      ox,     0     )  # outer bottom
    femm.mi_drawline(ox,    0,      ox,     th    )  # outer left
    femm.mi_drawline(ox,    th,     tw,     th    )  # outer top
    femm.mi_drawline(tw,    th,     tw,     th-aw-tip)  # outer right top
    femm.mi_drawline(tw,    th-aw-tip, tw-aw, th-aw-tip)  # gap top
    femm.mi_drawline(tw-aw, th-aw-tip, tw-aw, th-aw)      # top tip inner right
    femm.mi_drawline(tw-aw, th-aw,  ox+aw,  th-aw )  # inner top
    femm.mi_drawline(ox+aw, th-aw,  ox+aw,  aw    )  # inner left vertical
    femm.mi_drawline(ox+aw, aw,     tw-aw,  aw    )  # inner bottom
    femm.mi_drawline(tw-aw, aw,     tw-aw,  aw+tip)  # bottom tip inner right
    femm.mi_drawline(tw-aw, aw+tip, tw,     aw+tip)  # gap bottom
    femm.mi_drawline(tw,    aw+tip, tw,     0     )  # outer right bottom

    # left coil
    femm.mi_drawline(coil_left_x1, coil_y1, coil_left_x1, coil_y2)
    femm.mi_drawline(coil_left_x1, coil_y1, coil_left_x2, coil_y1)
    femm.mi_drawline(coil_left_x1, coil_y2, coil_left_x2, coil_y2)

    # right coil
    femm.mi_drawline(coil_right_x2, coil_y1, coil_right_x2, coil_y2)
    femm.mi_drawline(coil_right_x1, coil_y1, coil_right_x2, coil_y1)
    femm.mi_drawline(coil_right_x1, coil_y2, coil_right_x2, coil_y2)

    # Pure Iron - Label 
    femm.mi_addblocklabel((ox + tw)/2, aw/2)
    femm.mi_selectlabel((ox + tw)/2, aw/2)
    femm.mi_setblockprop('Pure Iron', 1, 0, '<None>', 0, 0, 0)
    femm.mi_clearselected()

    # left coil - Label
    femm.mi_addblocklabel((coil_left_x1 + coil_left_x2)/2, th/2)
    femm.mi_selectlabel((coil_left_x1 + coil_left_x2)/2, th/2)
    femm.mi_setblockprop('Copper', 1, 0, 'Coil', 0, 0, 500)
    femm.mi_clearselected()

    # right coil - Label
    femm.mi_addblocklabel((coil_right_x1 + coil_right_x2)/2, th/2)
    femm.mi_selectlabel((coil_right_x1 + coil_right_x2)/2, th/2)
    femm.mi_setblockprop('Copper', 1, 0, 'Coil', 0, 0, -500)
    femm.mi_clearselected()

    # air gap - Label
    femm.mi_addblocklabel(tw - aw/2, th/2)
    femm.mi_selectlabel(tw - aw/2, th/2)
    femm.mi_setblockprop('Air', 1, 0, '<None>', 0, 0, 0)
    femm.mi_clearselected()

    femm.mi_makeABC()
    femm.mi_saveas('ccore.fem')
    femm.mi_analyze()
    femm.mi_loadsolution()

    # flux density at key points in the magnetic circuit
    B_left = femm.mo_getb(ox + aw/2, th/2)
    B_gap  = femm.mo_getb(tw - aw/2, th/2)
    print(f'Left arm flux density:  By={B_left[1]:.4f} T')
    print(f'Air gap flux density:   By={B_gap[1]:.4f} T')

    # Flux Linkage through the coil
    circuit = femm.mo_getcircuitproperties('Coil')
    flux_linkage = circuit[2]
    print(f'Flux linkage: {flux_linkage:.6f} Wb')

    femm.mo_close()
    femm.mi_close()
    femm.closefemm()

    return flux_linkage


if __name__ == '__main__':
    flux_linkage = run_ccore_simulation(gap=5.0)