#iptables --flush
#iptables -A FORWARD -s 192.168.10.129/29 -o eth0 -j ACCEPT
#iptables -A FORWARD -d 192.168.10.129/29 -o eth1 -j ACCEPT

#iptables --flush
#iptables --append FORWARD --in-interface eth0 -j ACCEPT
#iptables --append FORWARD --in-interface eth1 -j ACCEPT
#iptables --append FORWARD --in-interface eth2 -j ACCEPT
#iptables --append FORWARD --in-interface eth3 -j ACCEPT
#iptables --append FORWARD --in-interface eth4 -j ACCEPT
#iptables -A FORWARD -j ACCEPT
#echo 1 > /proc/sys/net/ipv4/ip_forward


iptables --flush
iptables -t nat --flush

iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
iptables -A FORWARD -i eth0 -o eth1 -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i eth0 -o eth2 -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i eth0 -o eth3 -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i eth0 -o eth4 -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i eth0 -o eth5 -m state --state RELATED,ESTABLISHED -j ACCEPT

iptables -A FORWARD -i eth1 -o eth0 -j ACCEPT
iptables -A FORWARD -i eth2 -o eth0 -j ACCEPT
iptables -A FORWARD -i eth3 -o eth0 -j ACCEPT
iptables -A FORWARD -i eth4 -o eth0 -j ACCEPT
iptables -A FORWARD -i eth5 -o eth0 -j ACCEPT
echo 1 > /proc/sys/net/ipv4/ip_forward
