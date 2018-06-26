
from ml_super_class import MLSuperClass
import config

class MLBaseClass(MLSuperClass):
    '''
    machine learning base class
    '''    
    def __init__(self):
        '''
        base class constructor
        '''
        super().__init__()
    
#     MNIST in CSV
#     https://pjreddie.com/projects/mnist-in-csv/
    def convert_gz_to_csv(self, image_file, label_file, out_file, row_number):
        try:
            f = open(image_file, "rb")
            o = open(label_file, "w")
            l = open(out_file, "rb")
            f.read(16)
            l.read(8)
            images = []
            for i in range(row_number):
                image = [ord(l.read(1))]
                for j in range(28*28):
                    image.append(ord(f.read(1)))
                images.append(image)        
            for image in images:
                o.write(",".join(str(pix) for pix in image) + "\n")
        except:
            self.print_exception_message()
        finally:
            f.close()
            o.close()
            l.close()