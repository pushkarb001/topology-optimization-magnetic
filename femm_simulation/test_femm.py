import femm

# open femm and create a new magnetostatics problem
femm.openfemm()
femm.newdocument(0)

# set up problem definition
# units in millimeters, planar problem, frequency 0 (static)
femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, 0, 30)

print("FEMM opened and problem created successfully")

femm.closefemm()
print("FEMM closed successfully")