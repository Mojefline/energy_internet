import pandapower as pp
import numpy as np

def create_energy_router(net, bus):
    # ایجاد باس‌های مجازی
    virtual_bus1 = pp.create_bus(net, vn_kv=12.66, name=f"ER Virtual Bus 1 - {bus+1}")
    virtual_bus2 = pp.create_bus(net, vn_kv=12.66, name=f"ER Virtual Bus 2 - {bus+1}")
    
    # ایجاد ترانسفورماتورهای ایده‌آل
    pp.create_transformer_from_parameters(net, hv_bus=bus, lv_bus=virtual_bus1,
                                          sn_mva=10, vn_hv_kv=12.66, vn_lv_kv=12.66,
                                          vkr_percent=0, vk_percent=0.001, pfe_kw=0, i0_percent=0,
                                          shift_degree=0, tap_side="hv", tap_neutral=0,
                                          tap_min=-10, tap_max=10, tap_step_degree=0.5, tap_pos=0,
                                          name="ER P-Control")
    
    pp.create_transformer_from_parameters(net, hv_bus=virtual_bus1, lv_bus=virtual_bus2,
                                          sn_mva=10, vn_hv_kv=12.66, vn_lv_kv=12.66,
                                          vkr_percent=0, vk_percent=0.001, pfe_kw=0, i0_percent=0,
                                          tap_side="hv", tap_neutral=1, tap_min=0.9, tap_max=1.1,
                                          tap_step_percent=0.625, tap_pos=0,
                                          name="ER Q-Control")
    
    # ایجاد خط با امپدانس بسیار کم
    pp.create_line_from_parameters(net, from_bus=virtual_bus2, to_bus=bus, length_km=0.001,
                                   r_ohm_per_km=0.01, x_ohm_per_km=0.01, c_nf_per_km=0, max_i_ka=10,
                                   name="ER Return Line")
    
    return virtual_bus1, virtual_bus2

def adjust_energy_router(net, bus, virtual_bus1, virtual_bus2, target_voltage=1.0, target_p_mw=0):
    p_trafo = net.trafo[net.trafo.name == f"ER P-Control"]
    q_trafo = net.trafo[net.trafo.name == f"ER Q-Control"]
    
    # تنظیم کنترل توان اکتیو
    p_mw = net.res_bus.loc[bus, 'p_mw']
    p_error = target_p_mw - p_mw
    shift_adjust = np.sign(p_error) * min(abs(p_error) * 2, 0.5)  # محدود کردن تغییرات به 0.5 درجه
    new_shift = p_trafo.shift_degree.values[0] + shift_adjust
    p_trafo.shift_degree = np.clip(new_shift, -10, 10)
    
    # تنظیم کنترل ولتاژ (توان راکتیو)
    v_pu = net.res_bus.loc[bus, 'vm_pu']
    v_error = target_voltage - v_pu
    tap_adjust = np.sign(v_error) * min(abs(v_error) * 5, 1)  # محدود کردن تغییرات به 1 پله
    new_tap = q_trafo.tap_pos.values[0] + tap_adjust
    q_trafo.tap_pos = np.clip(new_tap, q_trafo.tap_min.values[0], q_trafo.tap_max.values[0])

# استفاده از توابع
net = pp.create_empty_network()
bus = pp.create_bus(net, vn_kv=12.66, name="Main Bus")
virtual_bus1, virtual_bus2 = create_energy_router(net, bus)

# اجرای پخش بار و تنظیم روتر انرژی
pp.runpp(net)
adjust_energy_router(net, bus, virtual_bus1, virtual_bus2)



