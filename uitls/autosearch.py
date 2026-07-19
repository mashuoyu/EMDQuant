import copy
from re import L
import numpy as np
from pyparsing import line
import torch
from binary import high_order_residual,high_order_residual_mask_cal
from utils.mask import generate_structural_mask

def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2, dim=(-2, -1))
    return mse

def calculate_percentage_and_variance_original(weights, abs_weights, bin_edges):
    percentages = []
    variances = []
    accum_percentages = [0]
    total_elements = abs_weights.numel()
    for i in range(len(bin_edges) - 1):
        bin_mask = (abs_weights >= bin_edges[i]) & (abs_weights < bin_edges[i + 1])
        bin_weights = weights[bin_mask]
        percentages.append(bin_weights.numel() / total_elements * 100)
        accum_percentages.append(accum_percentages[-1] + percentages[-1])
        variances.append(torch.var(bin_weights))
    return percentages, variances, accum_percentages

'''
Include main method to search the rate for 2-bit salient data columns and the optimal split for 1-bit data
'''
def structural_searching(origin_matrix, up_lim, exbit, row_block):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')
    origin_matrix_block = origin_matrix.view(row_block, -1, origin_matrix.shape[1])
    true_counts_block = origin_matrix_block.abs().sum(dim=1)
    block_row_size = int(origin_matrix.shape[0] / row_block)
    error = []
    lines = []
    optimal_split_0 = torch.zeros(1,row_block)
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns_indice = torch.topk(true_counts_block.flatten(), up_lim*row_block)
    top_braq_2_columns_indice_row = top_braq_2_columns_indice // true_counts_block.size(1) 
    top_braq_2_columns_indice_col = top_braq_2_columns_indice % true_counts_block.size(1) 
    top_braq_2_columns_indice_list = list(zip(top_braq_2_columns_indice_row.tolist(), top_braq_2_columns_indice_col.tolist()))


    mask2_origin = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    for i in range(1, up_lim*row_block):

        row_st = int(top_braq_2_columns_indice_list[i][0]*block_row_size)
        col = int(top_braq_2_columns_indice_list[i][1])


        mask2_origin[row_st: row_st + block_row_size , col] = True

        group3 = high_order_residual(origin_matrix, mask2_origin, order=3)

        group4 = high_order_residual(origin_matrix, ~mask2_origin, order=1)
        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0)
        lines.append(i)
        for j in range(row_block):
            if (quantize_error_0 < minimal_value_0):
                minimal_value_0 = quantize_error_0
                optimal_split_0 = i

    mask2_temp = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)


    _, top_braq_2_columns_indice = torch.topk(true_counts_block.flatten(),  optimal_split_0)
    top_braq_2_columns_indice_row = top_braq_2_columns_indice // true_counts_block.size(1) 
    top_braq_2_columns_indice_col = top_braq_2_columns_indice % true_counts_block.size(1) 
    mask2_position_list = list(zip(top_braq_2_columns_indice_row.tolist(), top_braq_2_columns_indice_col.tolist()))

    for i in range(optimal_split_0):

        row_st = int(mask2_position_list[i][0]*block_row_size)
        col = int(mask2_position_list[i][1])


        mask2_temp[row_st: row_st + block_row_size , col] = True


    
    one_two_matrix_block = (origin_matrix *(~mask2_temp)).view(row_block, -1, origin_matrix.shape[1])
    one_two_matrix_block = torch.where(torch.isnan(one_two_matrix_block), torch.zeros_like(one_two_matrix_block), one_two_matrix_block)
    one_two_true_counts = one_two_matrix_block.abs().sum(dim=1)
 
    _, secend_braq_2_indice = torch.topk(one_two_true_counts.flatten(), exbit)
    secend_braq_2_indice_row = secend_braq_2_indice // one_two_true_counts.size(1) 
    secend_braq_2_indicce_col = secend_braq_2_indice % one_two_true_counts.size(1) 
    secend_braq_2_indice_list = list(zip(secend_braq_2_indice_row.tolist(), secend_braq_2_indicce_col.tolist()))

    # search for the optimal split for the first, second and thrid group, high order=1 2 3, structured search


    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
   
    one_to_two_num = 0
    two_to_three_num = 0
    for i in range(exbit):

        quantize_2_to_3_error = []
        quantize_1_to_2_error = []

        if len(mask2_position_list):

            for j in range(min(exbit,len(mask2_position_list))):

                row_st = int(mask2_position_list[j][0]*block_row_size)
                col = int(mask2_position_list[j][1])

                mask2_temp[row_st: row_st + block_row_size , col]= False
                mask3[row_st: row_st + block_row_size , col]= True
                group1 = high_order_residual(origin_matrix, ~(mask2_temp| mask3) , order=1)
                group2 = high_order_residual(origin_matrix, mask2_temp, order=3)
                group3 = high_order_residual(origin_matrix, mask3, order=5)
                quantize_error = error_computing(origin_matrix, group1+group2+group3)
                quantize_2_to_3_error.append(quantize_error)
                mask2_temp[row_st: row_st + block_row_size , col]= True
                mask3[row_st: row_st + block_row_size , col]= False
        
            index_2_to_3 = quantize_2_to_3_error.index(min(quantize_2_to_3_error))

        for j in range(exbit - one_to_two_num):

            row_st = int(secend_braq_2_indice_list[j][0]*block_row_size)
            col = int(secend_braq_2_indice_list[j][1])

            mask2_temp[row_st: row_st + block_row_size , col]= True
            group1 = high_order_residual(origin_matrix, ~(mask2_temp| mask3) , order=1)
            group2 = high_order_residual(origin_matrix, mask2_temp, order=3)
            group3 = high_order_residual(origin_matrix, mask3, order=5)
            quantize_error = error_computing(origin_matrix, group1+group2+group3)
            quantize_1_to_2_error.append(quantize_error)
            mask2_temp[row_st: row_st + block_row_size , col]= False

        index_1_to_2 = quantize_1_to_2_error.index(min(quantize_1_to_2_error))

        if len(mask2_position_list) & (min(quantize_2_to_3_error) < min(quantize_1_to_2_error)):

            row_st = int(mask2_position_list[index_2_to_3][0]*block_row_size)
            col = int(mask2_position_list[index_2_to_3][1])
            
            mask3[row_st: row_st + block_row_size , col]= True
            mask2_temp[row_st: row_st + block_row_size , col]= False
            mask2_position_list.pop(index_2_to_3)
            two_to_three_num += 1
   

        else:
            row_st = int(secend_braq_2_indice_list[index_1_to_2][0]*block_row_size)
            col = int(secend_braq_2_indice_list[index_1_to_2][1])

            mask2_temp[row_st: row_st + block_row_size , col]= True
            mask2_position_list.append(secend_braq_2_indice_list[index_1_to_2])
            secend_braq_2_indice_list.pop(index_1_to_2)
            one_to_two_num += 1
            
    
    return ~(mask2_temp | mask3), mask2_temp, mask3

def structural_block_searching(origin_matrix, up_lim, exbit, row_block):
    minimal_value = float('inf')
    minimal_value_0 = torch.full((1,row_block), float('inf')).to("cuda")
    origin_matrix_block = origin_matrix.view(row_block, -1, origin_matrix.shape[1])
    true_counts_block = origin_matrix_block.abs().sum(dim=1)

    error = []
    lines = []
    optimal_split_0 = torch.zeros(1,row_block)
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts_block, up_lim)
    for i in range(1, up_lim):
        mask2_origin = torch.full((origin_matrix_block.shape[0], origin_matrix_block.shape[1], origin_matrix_block.shape[2]), False).to(origin_matrix.device)
        for j in range(row_block):
            mask2_origin[j,:, top_braq_2_columns[j,:i]] = True

        group3 = high_order_residual_mask_cal(origin_matrix_block, mask2_origin, order=3)

        group4 = high_order_residual_mask_cal(origin_matrix_block, ~mask2_origin, order=1)
        quantize_error_0 = error_computing(origin_matrix_block, group4+group3)
        error.append(quantize_error_0)
        lines.append(i)
        for j in range(row_block):
            if (quantize_error_0[j].item() < minimal_value_0[0,j].item()):
                minimal_value_0[0,j] = quantize_error_0[j]
                optimal_split_0[0,j] = i

    mask2_origin = torch.full((origin_matrix_block.shape[0], origin_matrix_block.shape[1], origin_matrix_block.shape[2]), False).to(origin_matrix.device)
    for j in range(row_block):
        _, top_braq_2_columns = torch.topk(true_counts_block[j], int(optimal_split_0[0,j]))
        mask2_origin[j,:, top_braq_2_columns] = True
        
    mask2_temp = copy.deepcopy(mask2_origin)
    
    one_two_matrix_block = origin_matrix_block * (~mask2_origin)
    one_two_true_counts = one_two_matrix_block.abs().sum(dim=1)

    _, secend_braq_2_columns_list = torch.topk(one_two_true_counts, exbit)

    # search for the optimal split for the first, second and thrid group, high order=1 2 3, structured search
    
    secend_braq_2_columns_list  = secend_braq_2_columns_list.cpu().numpy().tolist()

    mask3 = torch.full((origin_matrix_block.shape[0], origin_matrix_block.shape[1], origin_matrix_block.shape[2]), False).to(origin_matrix.device)

    
    
    for k in range(row_block):
        one_to_two_num = 0
        _, top_braq_2_columns = torch.topk(true_counts_block[k], int(optimal_split_0[0,k]))
        top_braq_2_columns = top_braq_2_columns.cpu().numpy().tolist()
        secend_braq_2_columns = secend_braq_2_columns_list[k]
        for i in range(exbit):

            quantize_2_to_3_error = []
            quantize_1_to_2_error = []

            if len(top_braq_2_columns):

                for j in range(min(exbit,int(optimal_split_0[0,k]))):
                    mask2_temp[k, :,top_braq_2_columns[j]]= False
                    mask3[k, :, top_braq_2_columns[j]]= True
                    group1 = high_order_residual(origin_matrix_block[k], ~(mask2_temp[k] | mask3[k]) , order=1)
                    group2 = high_order_residual(origin_matrix_block[k], mask2_temp[k], order=3)
                    group3 = high_order_residual(origin_matrix_block[k], mask3[k], order=5)
                    quantize_error = error_computing(origin_matrix_block[k], group1+group2+group3)
                    quantize_2_to_3_error.append(quantize_error)
                    mask2_temp[k,:,top_braq_2_columns[j]]= True
                    mask3[k,:,top_braq_2_columns[j]]= False
        
                index_2_to_3 = quantize_2_to_3_error.index(min(quantize_2_to_3_error))

            for l in range(exbit - one_to_two_num):
                mask2_temp[k,:,secend_braq_2_columns[l]]= True
                group1 = high_order_residual(origin_matrix_block[k], ~(mask2_temp[k] | mask3[k]), order=1)
                group2 = high_order_residual(origin_matrix_block[k], mask2_temp[k], order=3)
                group3 = high_order_residual(origin_matrix_block[k], mask3[k], order=5)
                quantize_error = error_computing(origin_matrix_block[k], group1+group2+group3)
                quantize_1_to_2_error.append(quantize_error)
                mask2_temp[:,secend_braq_2_columns[l]]= False

            index_1_to_2 = quantize_1_to_2_error.index(min(quantize_1_to_2_error))

            if len(top_braq_2_columns) & (min(quantize_2_to_3_error) < min(quantize_1_to_2_error)):
                mask3[k,:,top_braq_2_columns[index_2_to_3]]= True
                mask2_temp[k,:,top_braq_2_columns[index_2_to_3]]= False
                top_braq_2_columns.pop(index_2_to_3)
                optimal_split_0[0,k] -= 1

            else:
                mask2_temp[k,:,secend_braq_2_columns[index_1_to_2]]= True
                top_braq_2_columns.append(secend_braq_2_columns[index_1_to_2])
                secend_braq_2_columns.pop(index_1_to_2)
                one_to_two_num += 1
                optimal_split_0[0,k] += 1 
    
    return ~(mask2_temp | mask3).view(-1, origin_matrix.shape[1]), mask2_temp.view(-1, origin_matrix.shape[1]), mask3.view(-1, origin_matrix.shape[1])

def find_optimal_split(group_max, origin_matrix, border):
    optimal_split = None
    minimal_value = float('inf')
    searching_steps = torch.arange(0.1,0.8,0.01)
    searching_steps = searching_steps * group_max

    group3 = high_order_residual(origin_matrix, torch.abs(origin_matrix) > border, order=2)
    for split_value in searching_steps:

        group1 = high_order_residual(origin_matrix, (torch.abs(origin_matrix) > split_value) & (torch.abs(origin_matrix) <= border), order=1)
        group2 = high_order_residual(origin_matrix, torch.abs(origin_matrix) <= split_value, order=1)

        quantize_error = error_computing(origin_matrix, group1+group2+group3)
        if quantize_error < minimal_value:
            minimal_value = quantize_error
            optimal_split = split_value

    return optimal_split, minimal_value
