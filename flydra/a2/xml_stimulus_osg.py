import xml_stimulus
import contextlib,os,tempfile,shutil,math,sets
import numpy

import fsee
import fsee.scenegen.primlib as primlib
import fsee.scenegen.osgwriter as osgwriter

def make_floor( x0y0, x1y1, xtilesize=1, ytilesize=1, z=0 ):
    xo, yo = x0y0
    x1,y1 = x1y1
    xlen = x1-xo
    ylen = y1-yo
    xg = xtilesize
    yg = ytilesize
    maxi = int(math.ceil(xlen/xg))
    maxj = int(math.ceil(ylen/yg))
    floor = primlib.Prim()
    floor.texture_fname = "nearblack.png"
    count = 0
    quads = []
    for i in range(maxi):
        for j in range(maxj):
            floor.verts.append( [xo+i*xg,     yo+j*yg,     z] )
            floor.verts.append( [xo+i*xg,     yo+(j+1)*yg, z] )
            floor.verts.append( [xo+(i+1)*xg, yo+(j+1)*yg, z] )
            floor.verts.append( [xo+(i+1)*xg, yo+j*yg,     z] )
            floor.normals.append( (0,0,1) )
            floor.normals.append( (0,0,1) )
            floor.normals.append( (0,0,1) )
            floor.normals.append( (0,0,1) )
            floor.tex_coords.append( [0,0] )
            floor.tex_coords.append( [0,1] )
            floor.tex_coords.append( [1,1] )
            floor.tex_coords.append( [1,0] )
            quads.append( [count, count+1, count+2, count+3] )
            count += 4
    floor.prim_sets = [primlib.Quads( quads )]
    return floor

def cylindrical_arena(info=None):
    assets = ['greenred.png',
              'redgreen.png',
              'nearblack.png']
    assets = [ os.path.join(fsee.data_dir,
                            'models','mamarama_checkerboard',
                            asset) for asset in assets]
    geom = []

    radius = info['diameter']/2.0
    height = info['height']
    assert numpy.allclose(info['axis'],(0,0,1))
    origin = info['origin']

    if 1:
        res = 72
        angles = numpy.linspace( 0.0, 360.0, res+1 )
        starts = angles[:-1]
        stops = angles[1:]

        D2R = math.pi/180.0
        start_x = radius*numpy.cos(starts*D2R)
        start_y = radius*numpy.sin(starts*D2R)
        stop_x = radius*numpy.cos(stops*D2R)
        stop_y = radius*numpy.sin(stops*D2R)
        geode = osgwriter.Geode(states=['GL_LIGHTING OFF'])

        for i in range(len(start_x)):
            x1 = start_x[i]; y1 = start_y[i]
            x2 = stop_x[i]; y2 = stop_y[i]

            wall = primlib.XZRect()
            if i%2==0:
                wall.texture_fname = 'redgreen.png'
            else:
                wall.texture_fname = 'greenred.png'
            wall.mag_filter = "NEAREST"
            z0 = 0
            z1 = height
            verts = numpy.array([[ x1, y1, z0],
                                 [ x1, y1, z1],
                                 [ x2, y2, z1],
                                 [ x2, y2, z0]])
            verts = verts + origin[numpy.newaxis,:]
            wall.verts = verts
            geom.append(wall.get_as_osg_geometry())
    floor = make_floor( (-2.0, -2.0),(2.0,2.0), z=z0 )
    ceil  = make_floor( (-2.0, -2.0),(2.0,2.0), z=z1 )
    geom.extend( [floor.get_as_osg_geometry(), ceil.get_as_osg_geometry()] )
    return geom, assets

def cylindrical_post( info=None ):
    geom = []

    assets = ['nearblack.png']
    assets = [ os.path.join(fsee.data_dir,
                            'models','mamarama_checkerboard',
                            asset) for asset in assets]

    verts=info['verts']
    diameter=info['diameter']
    radius = diameter/2.0

    verts = numpy.asarray(verts)
    assert verts.shape == (2,3) # two 3D vertices
    direction = verts[1,:]-verts[0,:]

    if direction[0] != 0 or direction[1] != 0:
        import warnings
        warnings.warn('Post ends are currently (wrongly) forced to be same Z coordinate.')

    height = direction[2]
    origin = verts[0]

    if 1:
        res = 32
        angles = numpy.linspace( 0.0, 360.0, res+1 )
        starts = angles[:-1]
        stops = angles[1:]

        D2R = math.pi/180.0
        start_x = radius*numpy.cos(starts*D2R)
        start_y = radius*numpy.sin(starts*D2R)
        stop_x = radius*numpy.cos(stops*D2R)
        stop_y = radius*numpy.sin(stops*D2R)
        geode = osgwriter.Geode(states=['GL_LIGHTING OFF'])

        for i in range(len(start_x)):
            x1 = start_x[i]; y1 = start_y[i]
            x2 = stop_x[i]; y2 = stop_y[i]

            wall = primlib.XZRect()
            wall.texture_fname = "nearblack.png"
            z0 = 0
            z1 = height
            verts = numpy.array([[ x1, y1, z0],
                                 [ x1, y1, z1],
                                 [ x2, y2, z1],
                                 [ x2, y2, z0]])
            verts = verts + origin[numpy.newaxis,:]
            wall.verts = verts
            geom.append(wall.get_as_osg_geometry())

    return geom, assets

class StimulusWithOSG(xml_stimulus.Stimulus):
    @contextlib.contextmanager
    def OSG_model_path(self):

        real_osg_fnames = []
        geode = osgwriter.Geode(states=['GL_LIGHTING OFF'])
        all_assets = []
        for child in self.root:
            if child.tag in ['multi_camera_reconstructor','valid_h5_times']:
                continue
            elif child.tag == 'cylindrical_arena':
                info = self._get_info_for_cylindrical_arena(child)
                geom_elements, this_assets = cylindrical_arena( info=info )
                for el in geom_elements:
                    geode.append( el )
            elif child.tag == 'cylindrical_post':
                info = self._get_info_for_cylindrical_post(child)
                geom_elements, this_assets = cylindrical_post( info=info )
                for el in geom_elements:
                    geode.append( el )
            elif child.tag == 'osg_model':
                real_osg_fnames.append(child.text)
                this_assets = []
                #raise NotImplementedError('')
            else:
                import warnings
                warnings.warn("Unknown node: %s"%child.tag)
                this_assets = []
            all_assets.extend( this_assets )

        if len(real_osg_fnames):
            assert len(all_assets)==0 #can only have real .osg file or generated
            for real_osg_fname in real_osg_fnames:
                yield real_osg_fname
            return

        all_assets = sets.Set(all_assets) # remove redundant copies
        m = osgwriter.MatrixTransform(numpy.eye(4))
        m.append(geode)

        g = osgwriter.Group()
        g.append(m)

        tmpdir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(tmpdir,'autogenerated_stimulus.osg')
            fd = open(model_path,'wb')
            g.save(fd)
            fd.close()

            for asset in all_assets:
                target = os.path.join( tmpdir, os.path.split(asset)[-1] )
                shutil.copy2(asset, target)
            yield model_path
        finally:
            shutil.rmtree(tmpdir)
