import processing

def plot_process(q):
    import core_analysis
    while 1:
        filename, obj_id = q.get()
        obj_ids, unique_obj_ids, is_mat_file, data_file, extra = core_analysis.get_data(filename)
        print 'read',obj_id

class QuickPlot:
    def __init__(self):
        self.q = processing.PipeQueue()#maxsize=10) # PosixQueue having problems!
        self.started = False
    def plot(self,filename,obj_id):
        self._ensure_started()
        self.q.put( (filename, obj_id) )
    def _ensure_started(self):
        if self.started:
            return
        p = processing.Process(target=plot_process, args=[self.q])
        p.setDaemon(True)
        p.start()
        self.started = True

qp = QuickPlot()

def quick_plot(filename,obj_id):
    print 'write',obj_id
    qp.plot( filename, obj_id )

