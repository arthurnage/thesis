from detector.adwin_buckets import AdwinRowBucketList
from math import log, sqrt, fabs


class AdWin:
    def __init__(self, delta=0.002, max_buckets=5, min_clock=32, min_win_len=10, min_sub_win_len=5):
        """
        Class implementing adaprive window algorithm.
        """
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_clock = min_clock
        self.min_win_len = min_win_len
        self.min_sub_win_len = min_sub_win_len

        # Time is used for comparison with min_clock parameter
        self.time = 0
        # Current window length
        self.window_len = 0
        # Sum of all values in the window
        self.window_sum = 0.0
        # Variance of all values in the window
        self.window_variance = 0.0

        # Count of bucket row within window
        self.bucket_row_count = 0
        # Init bucket list
        self.bucket_row_list = AdwinRowBucketList(self.max_buckets)
        self.change_detected = 0

    def set_input(self, value):
        """
        Adds a new data value and automatically detects a possible concept drift.
        """
        self.time += 1

        # Insert the new element
        self.insert_element(value)

        # Reduce window
        return self.reduce_window()

    def insert_element(self, value):
        """
        Creates a new bucket, inserts it into a bucket row which is the head of the bucket row list.
        This bucket row can be compressed when it reaches the maximum number of buckets.
        """
        # Insert the new bucket
        self.bucket_row_list.head.insert_bucket(value, 0)

        # Calculate the incremental variance
        incremental_variance = 0
        if self.window_len > 0:
            mean = self.window_sum / self.window_len
            incremental_variance = self.window_len * pow(value - mean, 2) / (self.window_len + 1)

        # Update statistic value
        self.window_len += 1
        self.window_variance += incremental_variance
        self.window_sum += value

        # Compress buckets if necessary
        self.compress_bucket_row()

    def compress_bucket_row(self):
        """
        Merges two buckets within one row into a next bucket row if maximum number of buckets is reached.
        """
        bucket_row = self.bucket_row_list.head
        bucket_row_level = 0
        while bucket_row is not None:
            # Merge buckets if row is full
            if bucket_row.bucket_count == self.max_buckets + 1:
                next_bucket_row = bucket_row.next_bucket_row
                if next_bucket_row is None:
                    # Add new bucket row and move to it
                    self.bucket_row_list.add_to_tail()
                    next_bucket_row = bucket_row.next_bucket_row
                    self.bucket_row_count += 1

                # Calculate number of bucket which will be compressed into next bucket row
                n_1 = pow(2, bucket_row_level)
                n_2 = pow(2, bucket_row_level)
                # Mean
                mean_1 = bucket_row.bucket_sum[0] / n_1
                mean_2 = bucket_row.bucket_sum[1] / n_2
                # Total
                next_total = bucket_row.bucket_sum[0] + bucket_row.bucket_sum[1]
                # Variance
                external_variance = n_1 * n_2 * pow(mean_1 - mean_2, 2) / (n_1 + n_2)
                next_variance = bucket_row.bucket_variance[0] + bucket_row.bucket_variance[1] + external_variance

                # Insert into next bucket row, meanwhile remove two original buckets
                next_bucket_row.insert_bucket(next_total, next_variance)

                # Compress those tow buckets
                bucket_row.compress_bucket(2)

                # Stop if the number of bucket within one row, which does not exceed limited
                if next_bucket_row.bucket_count <= self.max_buckets:
                    break
            else:
                break

            # Move to next bucket row
            bucket_row = bucket_row.next_bucket_row
            bucket_row_level += 1

    def reduce_window(self):
        """
        Detects a change from last of each bucket row, reduces the window if there is a concept drift.
        """
        is_changed = False
        if self.time % self.min_clock == 0 and self.window_len > self.min_win_len:
            is_reduced_width = True
            while is_reduced_width:
                is_reduced_width = False
                is_exit = False
                n_0, n_1 = 0, self.window_len
                sum_0, sum_1 = 0, self.window_sum

                # Start building sub windows from the tail of window (old bucket row)
                bucket_row = self.bucket_row_list.tail
                i = self.bucket_row_count
                while (not is_exit) and (bucket_row is not None):
                    for bucket_num in range(bucket_row.bucket_count):
                        # Iteration of last bucket row, or last bucket in one row
                        if i == 0 and bucket_num == bucket_row.bucket_count - 1:
                            is_exit = True
                            break

                        # Grow sub window 0, while reduce sub window 1
                        n_0 += pow(2, i)
                        n_1 -= pow(2, i)
                        sum_0 += bucket_row.bucket_sum[bucket_num]
                        sum_1 -= bucket_row.bucket_sum[bucket_num]
                        diff_value = (sum_0 / n_0) - (sum_1 / n_1)

                        # Minimum sub window length is matching
                        if n_0 > self.min_sub_win_len + 1 and n_1 > self.min_sub_win_len + 1:
                            # Remove oldest bucket if there is a concept drift
                            if self.reduce_expression(n_0, n_1, diff_value):
                                is_reduced_width, is_changed = True, True
                                if self.window_len > 0:
                                    n_0 -= self.delete_element()
                                    is_exit = True
                                    break

                    # Move to previous bucket row
                    bucket_row = bucket_row.previous_bucket_row
                    i -= 1
        return is_changed

    def reduce_expression(self, n_0, n_1, diff_value):
        """
        Calculates epsilon cut value.
        """
        # Harmonic mean of n0 and n1
        m = 1 / (n_0 - self.min_sub_win_len + 1) + 1 / (n_1 - self.min_sub_win_len + 1)
        d = log(2 * log(self.window_len) / self.delta)
        variance_window = self.window_variance / self.window_len
        epsilon_cut = sqrt(2 * m * variance_window * d) + 2 / 3 * m * d
        return fabs(diff_value) > epsilon_cut

    def delete_element(self):
        """
        Removes a bucket from tail of bucket row.
        """
        bucket_row = self.bucket_row_list.tail
        deleted_number = pow(2, self.bucket_row_count)
        self.window_len -= deleted_number
        self.window_sum -= bucket_row.bucket_sum[0]

        deleted_bucket_mean = bucket_row.bucket_sum[0] / deleted_number
        inc_variance = bucket_row.bucket_variance[0] + deleted_number * self.window_len * pow(
            deleted_bucket_mean - self.window_sum / self.window_len, 2
        ) / (deleted_number + self.window_len)

        self.window_variance -= inc_variance

        # Delete bucket from bucket row
        bucket_row.compress_bucket(1)
        # If bucket row is empty, remove it from the tail of window
        if bucket_row.bucket_count == 0:
            self.bucket_row_list.remove_from_tail()
            self.bucket_row_count -= 1

        return deleted_number
