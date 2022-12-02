GET_FREQUENT_ITEMS_KERNEL = """
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

    __kernel void get_frequents(__global int* items, __global int* database, __global int* group_transactions_length,
    __global int* group_transactions_start, __global int* v_start_index, __global int* v_lengths, int nr_transactions,
    float min_supp, __global int* v_results, __local int* local_buffer)
    {
        int gid_0 = get_local_id(0);
        int gid_1 = get_global_id(1);
        int i, j, id;
        int n = nr_transactions;
        int work_group_size = get_local_size(0);

        local_buffer[gid_0] = (int)0;

        for(id = group_transactions_start[gid_0]; 
        (id<group_transactions_start[gid_0]+group_transactions_length[gid_0]) && (id<nr_transactions); 
        id++)
        {    
            if(InterpolationSearch(database, v_start_index[id],
            (v_start_index[id] + v_lengths[id]) - 1, items[gid_1]) == true)
            {
                local_buffer[gid_0] += 1;   
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int reduction_threads = work_group_size/2;
        while(reduction_threads>0)
        {
            if(gid_0< reduction_threads)
            {
                local_buffer[gid_0] += local_buffer[reduction_threads + gid_0];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            reduction_threads /= 2;
        }

        if(gid_0 == 0)
        {
            int freq = local_buffer[gid_0];
            float supp = (float)freq/(float)nr_transactions;
            if(supp >= min_supp)
            {
                v_results[gid_1] = 1;
            }
        }
    }"""

FREQUENT_ITEM_SET_MINER = """
    #define FALSE 0
    #define TRUE 1

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

    void write_left_item(int gi_item, int gi_parser_db, int start_write_poz, int id_item, int start_poz, 
    int item_set_width, __global int* write_container, __global int* write_start, __global int* read_container)
    {
        if(gi_item == 0 && gi_parser_db == 0)
        {
            // write item_set from left
            int t;
            for(t=0; t<item_set_width; ++t)
            {
                write_container[write_start[start_write_poz+id_item] + t] = read_container[start_poz-item_set_width+t];
            }
        }
    }

    void write_new_item_set(int gi_item, int id_item, int start_poz, int start_write_poz,
     int nr_left_item_sets, int start_item, int width_item, int item_set_width, __global int* write_container,
      __global int* items, __global int* read_container, __global int* write_start)
    {
        int start_write_poz_2 = start_write_poz + nr_left_item_sets;

        // find poz in starts container which holds the start positions for write operation
        int poz_of_start_write = start_write_poz_2 + gi_item*nr_left_item_sets + id_item;

        // get position for write
        int poz_to_write = write_start[poz_of_start_write];

        // write
        int write_idx;
        int write_items_end = poz_to_write + width_item;

        // write node items
        int z;
        for(z=start_item; z<width_item+start_item; ++z)
        {
            write_container[poz_to_write + z - start_item] = items[z];
        }
        poz_to_write += z - start_item;

        // write left node items
        for(write_idx=0; write_idx<item_set_width; ++write_idx)
        {
            write_container[poz_to_write+write_idx] = read_container[start_poz-item_set_width+write_idx];
        }
    }

    __kernel void miner(__global int* database, __global int* group_transactions_length, 
    __global int* group_transactions_start, __global int* start_index,
    __global int* lengths, __global int* items, __global int* items_heights, __global int* items_widths,
    __global int* items_start_items, __global int* items_start_2d,
    __global int* write_container, __global int* write_start, __global int* write_start_2d,
    __global int* read_container, __global int* read_heights, __global int* read_widths,
    __global int* read_start_index, __global int* read_start_2d, 
    int nr_transactions, int nr_nodes, float min_supp, __local int* local_work_buffer)
    {
        int gi_node = get_global_id(1);
        int gi_item = get_global_id(0);
        int gi_parser_db = get_global_id(2);
        int work_group_size = get_local_size(2);  

        // don't forget about expansion - > ???

        // if there is a item to be processed then the work_group will have to do work, else it won't
        if(gi_item < items_heights[gi_node])
        {
            // -> save start position and width for current item; when parsing database search for all of them,
            // -> if all are found then search for left   
            int start_item = items_start_items[items_start_2d[gi_node] + gi_item];
            int width_item = items_widths[items_start_2d[gi_node] + gi_item];

            // get left position to read from
            int left = gi_node + 1;

            // do if left exists
            if(left <= nr_nodes)
            {

                int nr_left_item_sets = read_heights[left];
                __local int start_poz; // needs to be updated after each left item-set
                start_poz = read_start_index[left];
                int start_write_poz = write_start_2d[gi_node];
                __local int id_item;

                // parse database for each left item_set
                for(id_item=nr_left_item_sets-1; id_item>=0; )
                {
                    int i, j, is_frequent = FALSE;

                    // it is needed a start index for widths container also, because it is 2D
                    int item_set_width = read_widths[read_start_2d[left] + id_item];
                    local_work_buffer[gi_parser_db] = 0;

                    //parse database
                    for(i = group_transactions_start[gi_parser_db]; 
                    (i<group_transactions_start[gi_parser_db]+group_transactions_length[gi_parser_db]); 
                    i++)
                    {
                        // parse each element in transaction to find X
                        //for(j=0; j<lengths[i]; ++j)
                        //{
                                // if X is found in transaction i then search for others, else go to next line
                                // search on line i from k=0 to k=lengths[i]-1, for all items in left item-set
                                int k;
                                int count_items = 0;
                                int count_left = 0;

                                // check if current items are included in line
                                for(k=start_item; k<width_item+start_item; ++k)
                                {
                                    if(InterpolationSearch(database, start_index[i],
                                    start_index[i] + lengths[i] - 1, items[k]) == true)
                                    {
                                        count_items++;  
                                    }
                                    else
                                    {
                                        break;
                                    }
                                } 

                                // check if left item_set is included in current line
                                if(count_items == width_item)
                                {
                                    for(k=0; k<item_set_width; ++k)
                                    {
                                        if(InterpolationSearch(database, start_index[i],
                                        start_index[i] + lengths[i] - 1,
                                        read_container[start_poz- item_set_width +k]) == true)
                                        {
                                            count_left++;  
                                        }
                                        else
                                        {
                                            break;
                                        }
                                    }
                                }

                                // if item_set was entirely found then increment transaction counter
                                if(count_left == item_set_width)
                                {
                                    local_work_buffer[gi_parser_db] += 1;
                                }
                                //break; 
                        //}
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    // get nr of occurrences using parallel reduction algorithm
                    int reduction_threads = work_group_size/2;
                    while(reduction_threads>0)
                    {
                        if(gi_parser_db < reduction_threads)
                        {
                            local_work_buffer[gi_parser_db] += local_work_buffer[reduction_threads + gi_parser_db];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                        reduction_threads /= 2;
                    }

                    // write the current item_set
                    write_left_item(gi_item, gi_parser_db, start_write_poz, id_item, start_poz, item_set_width,
                                    write_container, write_start, read_container);

                    // calculate supp and check if it is >= than min_supp
                    if(gi_parser_db == 0)
                    {
                        // calculate support
                        int nr_of_occurrences = local_work_buffer[gi_parser_db];
                        float supp = (float)nr_of_occurrences/(float)nr_transactions;

                        if(supp >= min_supp)
                        {
                            // this item_set is frequent
                            is_frequent = TRUE;

                            // yes -> write to write_container, write also only the item-set
                            // write the new item_set
                            write_new_item_set(gi_item, id_item, start_poz, start_write_poz, nr_left_item_sets,
                                                start_item, width_item, item_set_width, write_container, items,
                                                read_container, write_start);
                        }
                    }

                    // save start_poz
                    int current_start_pos = start_poz;

                    if(gi_parser_db == 0)
                    {
                        //update start_poz
                        start_poz -= item_set_width;
                        id_item--;
                    }

                    if(is_frequent == TRUE)
                    {
                        // check if next item_sets contain all their elements in this one
                        // YES -> next item_set is also frequent
                        // NO -> then check it in database

                        int check_next = TRUE;

                        while(check_next==TRUE && id_item>=0)
                        {

                            int next_item_set_width = read_widths[read_start_2d[left] + id_item];
                            __private int n, m, count_elements = 0;
                            for(n=0;n<next_item_set_width;++n)
                            {
                                for(m=0;m<item_set_width;++m)
                                {
                                    if(read_container[start_poz-next_item_set_width+n] == read_container[current_start_pos-item_set_width+m])
                                    {
                                        count_elements++;
                                        break;
                                    }
                                }
                            }

                            if(count_elements == next_item_set_width)
                            {
                                // next is also frequent
                                check_next = TRUE;

                                // write the current item_set
                                write_left_item(gi_item, gi_parser_db, start_write_poz, id_item, start_poz,
                                                next_item_set_width, write_container, write_start, read_container);
                                // write the new item_set
                                write_new_item_set(gi_item, id_item, start_poz, start_write_poz, nr_left_item_sets,
                                                    start_item, width_item, next_item_set_width, write_container,
                                                    items, read_container, write_start);

                                //update start_poz
                                start_poz -= next_item_set_width;

                                //go check next
                                id_item--;
                            }
                            else
                            {
                                check_next = FALSE;
                            }
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                } 
            }
        }
    }"""