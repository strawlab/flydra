"""cluster.py - abstraction of cluster

works for starcluster
in future for static IMP cluster
"""

import starcluster.config, starcluster.cluster

class ClusterBase(object):
    pass

class StarCluster(ClusterBase):
    def __init__(self, config_fname):
        self._tag_name = 'fwa'
        self._sg_name = '@sc-%s'%self._tag_name
        self._config = starcluster.config.get_config(config_fname)
        self._config.load()
        self._aws = self._config.get_aws_credentials()
        self.name = repr(self._aws)
        self.name = 'Amazon AWS (access key %s)'%(
            self._aws.aws_access_key_id,
            )

    def _get_sc_tags(self):
        print 'getting groups'
        import sys
        sys.stdout.flush()
        starcluster_groups = starcluster.cluster.get_cluster_security_groups(self._config)
        print 'got groups'
        sys.stdout.flush()
        result = []
        for scg in starcluster_groups:
            print 'getting tag',scg.name
            sys.stdout.flush()
            tag = starcluster.cluster.get_tag_from_sg(scg.name)
            print 'got tag',tag
            #cl = get_cluster(tag, cfg)
            #master = cl.master_node
            result.append( tag )
        print 'got all tags'
        sys.stdout.flush()
        return result

    def is_running(self):
        return self._tag_name in self._get_sc_tags()

    def get_num_nodes(self):
        cl = starcluster.cluster.get_cluster(self._tag_name, self._config)
        return len(cl.nodes)

    def shutdown(self):
        ec2 = self._config.get_easy_ec2()
        cluster = ec2.get_security_group(self._sg_name)
        for node in cluster.instances():
            node.stop()
        cluster.delete()

    def start_n_nodes(self,n):
        cfg = self._config
        template = 'single'
        scluster = cfg.get_cluster_template(template, self._tag_name)
        scluster.update({'cluster_size': n})
        assert scluster.is_valid()
        scluster.start(create=True)
