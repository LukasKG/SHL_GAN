# -*- coding: utf-8 -*-
from xlsxwriter.exceptions import FileCreateError

import xlsxwriter
import numpy as np

from log import log
 
def check_file_writable(name):
    try: 
        workbook = xlsxwriter.Workbook(name)
        workbook.close()
    except FileCreateError as e:
        return False
    else:
        return True

def get_name(params,num=None):
    return 'results_%s%s.xlsx'%(params['name'],"" if not num else "_("+str(num)+")")

def save(params,param_1,param_1_lst,param_2,param_2_lst,results):
    assert results.shape[0] == len(param_1_lst)
    assert results.shape[1] == len(param_2_lst)
    assert results.shape[2] == 3
    offset = results.shape[0] + 3
    
    name = get_name(params)
    i=0
    while not check_file_writable(name):
        log("Cannot access excel file \'%s\'"%name,name=params['log_name'])
        i += 1
        name = get_name(params,i)
        
    workbook = xlsxwriter.Workbook(name)
    worksheet = workbook.add_worksheet()
    
    b_size = 2
    form_def    = workbook.add_format({'valign': 'vcenter', 'align': 'center', 'num_format': '0.0000'})
    form_def_r  = workbook.add_format({'valign': 'vcenter', 'align': 'center', 'num_format': '0.0000', 'right': b_size})
    form_def_b  = workbook.add_format({'valign': 'vcenter', 'align': 'center', 'num_format': '0.0000', 'bottom': b_size})
    form_def_rb = workbook.add_format({'valign': 'vcenter', 'align': 'center', 'num_format': '0.0000', 'right': b_size, 'bottom': b_size})
    
    form_h_r    = workbook.add_format({'valign': 'vcenter', 'align': 'center', 'bg_color': 'E0E0E0', 'right': b_size})
    form_h_b    = workbook.add_format({'valign': 'vcenter', 'align': 'center', 'bg_color': 'E0E0E0', 'bottom': b_size})
    form_h_rb   = workbook.add_format({'valign': 'vcenter', 'align': 'center', 'bg_color': 'E0E0E0', 'right': b_size, 'bottom': b_size})
    
    form_black  = workbook.add_format({'valign': 'vcenter', 'align': 'center', 'bg_color': 'black', 'border': b_size})
    form_h_1    = workbook.add_format({'valign': 'vcenter', 'align': 'center', 'bg_color': 'A0A0A0', 'top': b_size, 'right': b_size, 'bottom': 1})
    form_h_2    = workbook.add_format({'valign': 'vcenter', 'align': 'center', 'bg_color': 'A0A0A0', 'right': 1, 'bottom': b_size, 'left': b_size, 'rotation': 90})
    
    worksheet.write(1, 1 + offset * 0,  'Performance Classifier')
    worksheet.write(1, 1 + offset * 1,  'Performance Generator')
    worksheet.write(1, 1 + offset * 2,  'Performance Discriminator')
    
         
    for o in range(results.shape[2]):
        # Data
        col = 3
        for x in range(results.shape[0]):
            row = 4
            for y in range(results.shape[1]):
                if x == results.shape[0]-1:
                    if y == results.shape[1]-1:
                        form = form_def_rb
                    else:
                        form = form_def_r
                else:
                    if y == results.shape[1]-1:
                        form = form_def_b
                    else:
                        form = form_def
                
                worksheet.write(row, col + offset*o,  results[x,y,o], form)
                row += 1
            col += 1
        
        # Black area
        worksheet.merge_range(2, 1 + offset*o, 3, 2 + offset*o, '', form_black)
       
        # Heatmap
        worksheet.conditional_format(4, 3 + offset*o, 4 + results.shape[1], 3 + results.shape[0] + offset*o,{'type': '3_color_scale'})
        
        # Column headings
        worksheet.merge_range(2, 3 + offset*o, 2, 2 + offset*o + results.shape[0], param_1, form_h_1)
        for x in range(results.shape[0]):
            worksheet.write(3, 3 + x + offset*o, param_1_lst[x], form_h_rb if x == results.shape[0]-1 else form_h_b)
     
        # Row headings
        worksheet.merge_range(4, 1 + offset*o, 3 + results.shape[1], 1 + offset*o, param_2, form_h_2)
        for y in range(results.shape[1]):
            worksheet.write(4 + y, 2 + offset*o, param_2_lst[y], form_h_rb if y == results.shape[1]-1 else form_h_r)
        
    workbook.close()
    
if __name__ == "__main__":
    res = np.random.random(size=(5,4,3))
    print(res)
    
    par = {}
    par['name'] = 'test'
    par['log_name'] = 'log'
    
    save(par,'param_1',np.random.random((res.shape[0])),'param_2',np.random.random((res.shape[1])),res)