from __future__ import with_statement
import contextlib, warnings
import tables
import numpy as np
import os
import tempfile


def clear_col(dest_table, colname, fill_value=np.nan):
    if 0:
        objcol = dest_table._get_column_instance(colname)
        descr = [objcol._v_parent._v_nested_descr[objcol._v_pos]]
        dtype = descr[0][1]

        nancol = np.ones((dest_table.nrows,), dtype=dtype)
        # recarray = np.rec.array( nancol, dtype=descr)

        dest_table.modify_column(column=nancol, colname="x")
        dest_table.flush()
    else:
        warnings.warn("slow implementation of column clearing")
        for row in dest_table:
            row[colname] = fill_value
            row.update()


@contextlib.contextmanager
def open_file_safe(filename, delete_on_error=False, **kwargs):
    """open a file that will be closed when it goes out of scope

    This is very similar to contextlib.closing(), but optionally
    deletes file on error.
    """
    if delete_on_error:
        if os.path.exists(filename):
            raise RuntimeError("will not overwrite exiting file %r" % filename)
        # create filename in appropriate directory
        out_dir = os.path.split(os.path.abspath(filename))[0]
        f = tempfile.NamedTemporaryFile(dir=out_dir, delete=False)
        use_fname = f.name
        f.close()
        del f
    else:
        use_fname = filename

    result = tables.open_file(use_fname, **kwargs)
    try:
        yield result
    except:
        result.close()
        if delete_on_error:
            os.unlink(use_fname)
        raise
    finally:
        result.close()

    if delete_on_error:
        # We had no error if we are here, so the file is OK.
        os.rename(use_fname, filename)
