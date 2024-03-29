from .adwin_bucket_row import AdwinBucketRow


class AdwinRowBucketList:
    """
    A list of bucket rows.
    Class allows to add new bucket row at head of window,
    and remove old bucket row from tail of window.
    """
    def __init__(self, max_buckets=5):
        self.max_buckets = max_buckets

        self.count = 0
        self.head = None
        self.tail = None
        self.add_to_head()

    def add_to_head(self):
        self.head = AdwinBucketRow(self.max_buckets, next_bucket_row=self.head)
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def add_to_tail(self):
        """
        Add the bucket row at the end of the window.
        """
        self.tail = AdwinBucketRow(self.max_buckets, previous_bucket_row=self.tail)
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def remove_from_tail(self):
        """
        Remove the last bucket row in the window.
        """
        self.tail = self.tail.previous_bucket_row
        if self.tail is None:
            self.head = None
        else:
            self.tail.next_bucket_row = None
        self.count -= 1
