class PageHinkley:
    """PH test"""
    def __init__(self, delta=0.005, lmbda=50, alpha=1 - 0.001):
        self.delta = delta
        self.lmbda = lmbda
        self.alpha = alpha
        self.sum = 0
        self.x_mean = 0
        self.num = 0
        self.change_detected = False


    def reset_params(self):
        """
        Resets collected statistics
        """
        self.num = 0
        self.x_mean = 0
        self.sum = 0

    def set_input(self, x):
        """
        Adds a new value and detects a concept drift
        """
        self.detect_drift(x)
        return self.change_detected

    def detect_drift(self, x):
        # calculates the average and sum
        self.num += 1
        self.x_mean = (x + self.x_mean * (self.num - 1)) / self.num
        self.sum = self.sum * self.alpha + (x - self.x_mean - self.delta)

        self.change_detected = True if self.sum > self.lmbda else False
        if self.change_detected:
            self.reset_params()
