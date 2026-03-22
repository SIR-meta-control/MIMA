# msg for sync read instruction
# 1: {torque enable, present current,
#     present velocity, present position,
#     input voltage, temperature} 
# 2: Motor ID list 
# 3: Param list
# ---
# res: Param list
uint8 paramType
uint8[] motorID
uint32[] params