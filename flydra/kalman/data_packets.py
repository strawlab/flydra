import struct

packet_header_fmt = '<idBB' # XXX check format
packet_header_fmtsize = struct.calcsize(packet_header_fmt)

super_packet_header_fmt = '<H'
super_packet_header_fmtsize = struct.calcsize(super_packet_header_fmt)
super_packet_subheader = 'H'

err_size = 1

def decode_data_packet(buf):
    # keep in sync with flydra_tracker.py's Tracker.encode_data_packet()
    header = buf[:packet_header_fmtsize]
    rest = buf[packet_header_fmtsize:]

    (corrected_framenumber,timestamp,N,state_size) = struct.unpack(
        packet_header_fmt,header)
    per_tracked_object_fmt = 'f'*(state_size+err_size)
    per_tracked_object_fmtsize = struct.calcsize(per_tracked_object_fmt)
    state_vecs = []
    for i in range(N):
        this_tro = rest[:per_tracked_object_fmtsize]
        rest = rest[per_tracked_object_fmtsize:]

        results = struct.unpack(per_tracked_object_fmt,this_tro)
        state_vec = results[:state_size]
        meanP = results[state_size]
        state_vecs.append( state_vec )
    return corrected_framenumber, timestamp, state_vecs, meanP

def encode_super_packet( data_packets ):
    # These data packets come from flydra_tracker.Tracker.encode_data_packet()
    n = len(data_packets)
    sizes = [ len(p) for p in data_packets ]
    fmt = super_packet_header_fmt + (super_packet_subheader)*n
    super_packet_header = struct.pack( fmt, n, *sizes )
    final_packet = super_packet_header + ''.join(data_packets)
    return final_packet

def decode_super_packet( super_packet ):
    header = super_packet[:super_packet_header_fmtsize]
    rest = super_packet[super_packet_header_fmtsize:]

    (n,) = struct.unpack(super_packet_header_fmt,header)
    fmt2 = (super_packet_subheader)*n
    fmt2size = struct.calcsize(fmt2)

    subheader = rest[:fmt2size]
    data_packets_joined = rest[fmt2size:]
    sizes = struct.unpack( fmt2, subheader )

    data_packets = []
    next_packets = data_packets_joined

    for sz in sizes:
        this_packet = next_packets[:sz]
        next_packets = next_packets[sz:]

        data_packets.append( this_packet )
    return data_packets
