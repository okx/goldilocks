#include "ff/goldilocks.hpp"

const fr_t group_gen = fr_t(0x0000000000000007);
const fr_t group_gen_inverse = fr_t(0x249249246db6db6e);

const int S = 32;



const fr_t forward_roots_of_unity[S + 1] = {
        fr_t(0x1),
        fr_t(18446744069414584320ULL),
        fr_t(281474976710656ULL),
        fr_t(16777216ULL),
        fr_t(4096ULL),
        fr_t(64ULL),
        fr_t(8ULL),
        fr_t(2198989700608ULL),
        fr_t(4404853092538523347ULL),
        fr_t(6434636298004421797ULL),
        fr_t(4255134452441852017ULL),
        fr_t(9113133275150391358ULL),
        fr_t(4355325209153869931ULL),
        fr_t(4308460244895131701ULL),
        fr_t(7126024226993609386ULL),
        fr_t(1873558160482552414ULL),
        fr_t(8167150655112846419ULL),
        fr_t(5718075921287398682ULL),
        fr_t(3411401055030829696ULL),
        fr_t(8982441859486529725ULL),
        fr_t(1971462654193939361ULL),
        fr_t(6553637399136210105ULL),
        fr_t(8124823329697072476ULL),
        fr_t(5936499541590631774ULL),
        fr_t(2709866199236980323ULL),
        fr_t(8877499657461974390ULL),
        fr_t(3757607247483852735ULL),
        fr_t(4969973714567017225ULL),
        fr_t(2147253751702802259ULL),
        fr_t(2530564950562219707ULL),
        fr_t(1905180297017055339ULL),
        fr_t(3524815499551269279ULL),  // rou_31, where  rou_32^{1<<31} = 1
        fr_t(7277203076849721926ULL),  // rou_32, where  rou_32^{1<<32} = 1
};

const fr_t inverse_roots_of_unity[S + 1] = {
    fr_t(0x0000000000000001),
    fr_t(0xffffffff00000000),
    fr_t(0xfffeffff00000001),
    fr_t(0x000000ffffffff00),
    fr_t(0x0000001000000000),
    fr_t(0xfffffffefffc0001),
    fr_t(0xfdffffff00000001),
    fr_t(0xffefffff00000011),
    fr_t(0x1d62e30fa4a4eeb0),
    fr_t(0x3de19c67cf496a74),
    fr_t(0x3b9ae9d1d8d87589),
    fr_t(0x76a40e0866a8e50d),
    fr_t(0x9af01e431fbd6ea0),
    fr_t(0x3712791d9eb0314a),
    fr_t(0x409730a1895adfb6),
    fr_t(0x158ee068c8241329),
    fr_t(0x6d341b1c9a04ed19),
    fr_t(0xcc9e5a57b8343b3f),
    fr_t(0x22e1fbf03f8b95d6),
    fr_t(0x46a23c48234c7df9),
    fr_t(0xef8856969fe6ed7b),
    fr_t(0xa52008ac564a2368),
    fr_t(0xd46e5a4c36458c11),
    fr_t(0x4bb9aee372cf655e),
    fr_t(0x10eb845263814db7),
    fr_t(0xc01f93fc71bb0b9b),
    fr_t(0xea52f593bb20759a),
    fr_t(0x91f3853f38e675d9),
    fr_t(0x3ea7eab8d8857184),
    fr_t(0xe4d14a114454645d),
    fr_t(0xe2434909eec4f00b),
    fr_t(0x95c0ec9a7ab50701),
    fr_t(0x76b6b635b6fc8719),  // rou_inv_32, where rou_inv_32 = rou_32^{-1}
};

const fr_t domain_size_inverse[S + 1] = {
    fr_t(0x0000000000000001),  // 1^{-1}
    fr_t(0x7fffffff80000001),  // 2^{-1}
    fr_t(0xbfffffff40000001),  // (1 << 2)^{-1}
    fr_t(0xdfffffff20000001),  // (1 << 3)^{-1}
    fr_t(0xefffffff10000001),
    fr_t(0xf7ffffff08000001),
    fr_t(0xfbffffff04000001),
    fr_t(0xfdffffff02000001),
    fr_t(0xfeffffff01000001),
    fr_t(0xff7fffff00800001),
    fr_t(0xffbfffff00400001),
    fr_t(0xffdfffff00200001),
    fr_t(0xffefffff00100001),
    fr_t(0xfff7ffff00080001),
    fr_t(0xfffbffff00040001),
    fr_t(0xfffdffff00020001),
    fr_t(0xfffeffff00010001),
    fr_t(0xffff7fff00008001),
    fr_t(0xffffbfff00004001),
    fr_t(0xffffdfff00002001),
    fr_t(0xffffefff00001001),
    fr_t(0xfffff7ff00000801),
    fr_t(0xfffffbff00000401),
    fr_t(0xfffffdff00000201),
    fr_t(0xfffffeff00000101),
    fr_t(0xffffff7f00000081),
    fr_t(0xffffffbf00000041),
    fr_t(0xffffffdf00000021),
    fr_t(0xffffffef00000011),
    fr_t(0xfffffff700000009),
    fr_t(0xfffffffb00000005),
    fr_t(0xfffffffd00000003),
    fr_t(0xfffffffe00000002),  // (1 << 32)^{-1}
};
