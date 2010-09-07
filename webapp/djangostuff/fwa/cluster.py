"""cluster.py - abstraction of cluster

works for starcluster
in future for static IMP cluster
"""

import starcluster.config, starcluster.cluster
import starcluster.addnode
import starcluster.removenode
from flydra.sge_utils.util import COUCHDB_STATUS_DOC_ID
from couchdb.client import Server
import couchdb.http
import pytz, datetime

class ClusterBase(object):

    def update_couch_view_of_sge_status(self, couch_url, db_name):
        if self.is_running():
            cmd = '/home/astraw/PY/bin/flydra_sge_update_couch_view_of_sge_status %s %s'%(couch_url, db_name)
            return self._execute_on_master(cmd)
        else:
            couch_server = Server(couch_url)
            db = couch_server[db_name]
            doc = {'_id':COUCHDB_STATUS_DOC_ID}
            try:
                orig_doc = db[COUCHDB_STATUS_DOC_ID]
                doc['_rev'] = orig_doc['_rev']
            except couchdb.http.ResourceNotFound:
                pass
            doc['update_time'] =  pytz.utc.localize( datetime.datetime.utcnow() ).isoformat()
            doc['status']='stopped'
            db.update( [doc] )

class StarCluster(ClusterBase):
    def __init__(self, config_fname):
        self._template = 'single'
        self._tag_name = 'fwa'
        self._sg_name = '@sc-%s'%self._tag_name
        self._config = starcluster.config.get_config(config_fname)
        self._config.load()
        self._aws = self._config.get_aws_credentials()
        self.name = repr(self._aws)
        self.name = 'Amazon AWS (access key %s), StarCluster template "%s", tag "%s"'%(
            self._aws.aws_access_key_id,
            self._template, self._tag_name,
            )

    def _get_sc_tags(self):
        starcluster_groups = starcluster.cluster.get_cluster_security_groups(self._config)
        result = []
        for scg in starcluster_groups:
            tag = starcluster.cluster.get_tag_from_sg(scg.name)
            result.append( tag )
        return result

    def is_running(self):
        return self._tag_name in self._get_sc_tags()

    def get_num_nodes(self):
        cl = starcluster.cluster.get_cluster(self._tag_name, self._config)
        return len(cl.nodes)

    def shutdown(self):
        cl = starcluster.cluster.get_cluster(self._tag_name, self._config)
        cl.stop_cluster()

    def start_n_nodes(self,n):
        cfg = self._config
        scluster = cfg.get_cluster_template(self._template, self._tag_name)
        scluster.update({'cluster_size': n})
        assert scluster.is_valid()
        scluster.start(create=True)

    def modify_num_nodes(self,n):
        cl = starcluster.cluster.get_cluster(self._tag_name, self._config)
        if n==0:
            return # do nothing
        if n>0:
            # add nodes
            starcluster.addnode.add_nodes(cl, n )
        else:
            assert n<0
            n = abs(n)
            raise NotImplementedError('cannot remove %d nodes: find out which node to remove'%n)
            starcluster.removenode.remove_node( node, cl)

    def _execute_on_master(self,cmd):
        cl = starcluster.cluster.get_cluster(self._tag_name, self._config)
        master = cl.master_node
        return master.ssh.execute("su -l -c '%s' astraw"%cmd)
