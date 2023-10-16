def pasInit(secs=None):
    from neuron import h

    h.finitialize(h.v_init)
    h.fcurrent()
    for sec in secs:
        mechs = sec.psection()["density_mechs"]
        sec_cur = (
            (sec.ik if hasattr(sec, "ik") else 0)
            + (sec.ina if hasattr(sec, "ina") else 0)
            + (sec.ica if hasattr(sec, "ica") else 0)
            + (sec.icl if hasattr(sec, "icl") else 0)
        )
        for seg in sec:
            cur = sec_cur
            for ion in ["i", "ihi"]:
                for k in mechs:
                    try:
                        cur += getattr(getattr(seg, mech), ion)
                    except:
                        pass
            seg.pas.e = h.v_init + cur / seg.pas.g
            #print(seg, seg.pas.e)
