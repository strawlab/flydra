from __future__ import with_statement
import contextlib, warnings
import tables
import numpy as np
import os

def clear_col(dest_table, colname, fill_value=np.nan):
    if 0:
        objcol = dest_table._getColumnInstance(colname)
        descr = [objcol._v_parent._v_nestedDescr[objcol._v_pos]]
        dtype = descr[0][1]

        nancol = np.ones( (dest_table.nrows,), dtype=dtype)
        #recarray = np.rec.array( nancol, dtype=descr)

        dest_table.modifyColumn(column=nancol, colname='x')
        dest_table.flush()
    else:
        warnings.warn('slow implementation of column clearing')
        for row in dest_table:
            row[colname] = fill_value
            row.update()

@contextlib.contextmanager
def openFileSafe(filename,delete_on_error=False,**kwargs):
    result = tables.openFile(filename,**kwargs)
    try:
        yield result
    except:
        result.close()
        if delete_on_error:
            os.unlink(filename)
        raise
    finally:
        result.close()
