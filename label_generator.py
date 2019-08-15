import math

class LabelGenerator:

    def __init__(self, label_names, val_range):
        """
        Creates a label object that maps a continuous range to a label
        :param label_names: List of labels
        :param val_range: Range of continuous variable (tuple or list length 2)
        """

        self.val_range = val_range
        self.label_names = label_names
        self.min_range = val_range[0]
        self.max_range = val_range[1]
    

    def value_to_label_index(self, val):
        """
        Returns the index of the mapped label
        :param val: Value of continuous variable (float)
        """

        # Range check
        if val <= self.min_range:
            return 0
        elif val >= self.max_range:
            return len(self.label_names) - 1

        bucket_size = float(self.max_range - self.min_range) / len(self.label_names)
        bucket_index = math.floor((val - self.min_range) / bucket_size)

        return bucket_index
        

    def value_to_label_name(self, val):
        """
        Returns the name of the mapped label
        :param val: Value of continuous variable (float)
        """
        return self.label_names[self.value_to_label_index(val)]


    def index_to_name(self, index):
        """
        Returns the name of the label given a mapped index
        :param index: Mapped index
        """
        return self.label_names[index]

    
    def get_label_names(self):

        return self.label_names


    def __repr__(self):

        return "Label object: {} mapped to {}".format(self.val_range, self.label_names)


if __name__ == '__main__':
    steering_label_names = ['left', 'half_left', 'neutral', 'half_right', 'right']
    steering_value_range = (-1, 1)

    label_generator = LabelGenerator(steering_label_names, steering_value_range)
    
    steering_val = 4

    name = label_generator.value_to_label_index(steering_val)
    print(name)