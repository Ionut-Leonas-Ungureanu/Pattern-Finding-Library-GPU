APRIORI_2D_KERNEL = """ 
        bool InterpolationSearch(__global int* database, int begin, int end, int elem) 
        {
            while (database[begin] <= elem && elem <= database[end] && begin != end)
            {
                int mid = begin + ((elem - database[begin]) * (end-begin) / (database[end] - database[begin]));
                   
                if (database[mid] == elem) 
                    return true;
                else if (database[mid] < elem)
                    begin = mid + 1;
                else
                    end = mid;
            }
            
            return false;
        }

        __kernel void process_item_sets(__global int *item_sets, __global int *database,
        __global int* group_transactions_length, __global int* group_transactions_start, int k, int nr_transactions, 
        float min_supp, __global int *v_start_index,__global int *v_lengths, __global int *v_frequency_provider,
        __local int* local_buffer)
        {
            int gid_0 = get_local_id(0);
            int gid_1 = get_global_id(1);
            int i, j, id;
            int count = 0;
            int n = nr_transactions;
            int work_group_size = get_local_size(0);

            local_buffer[gid_0] = 0;

            if(gid_0 < nr_transactions)
            {
                for(id = group_transactions_start[gid_0]; 
                (id<(group_transactions_start[gid_0]+group_transactions_length[gid_0])) && (id<nr_transactions); 
                id++)
                {
                    count = 0;
                    for(i=(gid_1 * k); i<(gid_1 * k) + k; ++i)
                    {
                        if(InterpolationSearch(database, v_start_index[id],
                        (v_start_index[id] + v_lengths[id]) - 1, item_sets[i]) == true)
                        {
                            count++;     
                        }
                        else
                        {
                            break;
                        }
                    }

                    if(count == k)
                    {
                        local_buffer[gid_0] += 1;                
                        count = 0;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            int reduction_threads = work_group_size/2;
            while(reduction_threads>0)
            {
                if(gid_0 < reduction_threads)
                {
                    local_buffer[gid_0] += local_buffer[reduction_threads + gid_0];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                reduction_threads /= 2;
            }

            if(gid_0 == 0)
            {
                int freq = local_buffer[gid_0];
                //printf("freq = %d\\n", freq);
                float supp = (float)freq/(float)nr_transactions;
                //printf("supp = %d\\n", gid_0);
                if(supp >= min_supp)
                {
                    v_frequency_provider[gid_1] = 1;
                }
            }
        }
        """

APRIORI_CANDIDATES_GENERATION = """
    int intersection_length(__local int* first, __local int* second,int n)
    {
        int count = 0;
        int i, j;
        for(i=0; i<n; i++)
        {
            for(j=0; j<n; j++)
            {
                if(first[i] == second[j])
                {
                    count++;
                    break;
                }
            }
        }
        return count;
    }

    int union_arrays(__local int* first,__local int* second, int n,__local int* candidate)
    {
        int i = 0, j = 0, k = 0;

        while(i<n && j<n)
        {
            if(first[i] < second[j])
            {
                candidate[k++] = first[i++];
            }
            if(first[i] > second[j])
            {
                candidate[k++] = second[j++];
            }
            if(first[i] == second[j])
            {
                candidate[k++] = first[i];
                i++;
                j++;
            }
        }

        while(i < n)
        {
            candidate[k++] = first[i++];
        }

        while(j < n)
        {
            candidate[k++] = second[j++];
        }
        
        return k;
    }

    __kernel void generate_candidates(__global int* item_sets, int nr_item_sets, int k, __global int* results, 
    __global int* starts_results, __local int* first, __local int* second, __local int* candidate)
    {
        int first_id = get_global_id(1);
        int second_id = get_global_id(0);

        if(second_id > first_id)
        {
            int feature_k = k + 1;

            int i;
            int start_read = first_id*k;
            int stop_read = first_id*k + k;
            for(i=start_read; i<stop_read; i++)
            {
                first[i-start_read] = item_sets[i];
            }

            start_read = second_id*k;
            stop_read = second_id*k + k;
            for(i=start_read; i<stop_read;i++)
            {
                second[i-start_read] = item_sets[i];
            }
            
            int virtual_k = union_arrays(first, second, k, candidate);

            if(virtual_k == k+1)
            {
                int start_write = starts_results[first_id];
                start_write = start_write + (second_id - first_id - 1)*feature_k;
    
                for(i=start_write; i<start_write+feature_k; i++)
                {
                    results[i] = candidate[i-start_write];
                }
            }
        }
    }
"""