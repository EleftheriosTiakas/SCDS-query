// ------------------------------------------------------------
//
//  main.cpp
//
//  This code includes the implementation of the
//  Subspace Consensus Dominating Skyline Algorithm (SCDS)
//  and a corresponding baseline algorithm (BF)
//
//  Created by Eleftherios Tiakas on 9/5/26
//  and updated on 16/7/26
//
// ------------------------------------------------------------



#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

// ------------------------------------------------------------
// Basic Structures
// ------------------------------------------------------------

struct Tuple
{
    int id = -1;
    vector<double> x;  // note that all attribute values must have a minimization preference
};

struct DataSet
{
    vector<Tuple> rows;
    int dim() const { return rows.empty() ? 0 : static_cast<int>(rows[0].x.size()); }
    size_t size() const { return rows.size(); }
};

struct Subspace
{
    vector<int> attrs;
};

// ------------------------------------------------------------
// Some Utility Functions
// ------------------------------------------------------------

static inline bool dominates_in_subspace(const Tuple& a, const Tuple& b, const Subspace& S)
{
    bool strict = false;
    for (int idx : S.attrs)
    {
        if (a.x[idx] > b.x[idx]) return false;
        if (a.x[idx] < b.x[idx]) strict = true;
    }
    return strict;
}

static inline double monotone_priority(const Tuple& t, const Subspace& S)
{
    double s = 0.0;
    for (int idx : S.attrs) s += t.x[idx];
    return s;
}

static inline string dist_name(int mode)
{
    if (mode == 0) return "uniform";
    if (mode == 1) return "correlated";
    return "anticorrelated";
}

// ------------------------------------------------------------
// Synthetic Data Generators
// ------------------------------------------------------------

DataSet make_uniform(int n, int d, uint64_t seed)
{
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0.0, 1.0);
    DataSet ds;
    ds.rows.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        Tuple t;
        t.id = i;
        t.x.resize(d);
        for (int j = 0; j < d; ++j) t.x[j] = U(rng);
        ds.rows.push_back(std::move(t));
    }
    return ds;
}

DataSet make_correlated(int n, int d, uint64_t seed)
{
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0.0, 1.0);
    normal_distribution<double> N(0.0, 0.03);
    DataSet ds;
    ds.rows.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        double base = U(rng);
        Tuple t;
        t.id = i;
        t.x.resize(d);
        for (int j = 0; j < d; ++j)
        {
            double v = base + N(rng);
            v = max(0.0, min(1.0, v));
            t.x[j] = v;
        }
        ds.rows.push_back(std::move(t));
    }
    return ds;
}

DataSet make_anticorrelated(int n, int d, uint64_t seed)
{
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0.0, 1.0);
    normal_distribution<double> N(0.0, 0.03);
    DataSet ds;
    ds.rows.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        double base = U(rng);
        Tuple t;
        t.id = i;
        t.x.resize(d);
        if (d >= 1) t.x[0] = base;
        for (int j = 1; j < d; ++j)
        {
            double v = 1.0 - base + N(rng);
            v = max(0.0, min(1.0, v));
            t.x[j] = v;
        }
        ds.rows.push_back(std::move(t));
    }
    return ds;
}

DataSet make_synthetic(int n, int d, const string& distribution, uint64_t seed)
{
    if (distribution == "uniform") return make_uniform(n, d, seed);
    if (distribution == "correlated") return make_correlated(n, d, seed);
    return make_anticorrelated(n, d, seed);
}

// ------------------------------------------------------------
// Functions for loading data from a CSV file
// Format:
//   1st line: header row
//   next lines: ID (or item-name) in first column,
//               numeric data in the rest columns
// ------------------------------------------------------------

vector<string> split_csv_line(const string& line)
{
    vector<string> out;
    string cur;
    bool in_quotes = false;
    for (char c : line)
    {
        if (c == '"') in_quotes = !in_quotes;
        else if (c == ',' && !in_quotes)
        {
            out.push_back(cur);
            cur.clear();
        }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

DataSet load_csv_numeric(const string& path, int skip_cols = 1)
{
    ifstream fin(path);
    if (!fin) throw runtime_error("Cannot open CSV: " + path);

    string line;
    getline(fin, line); // header

    DataSet ds;
    int row_id = 0;
    while (getline(fin, line))
    {
        if (line.empty()) continue;
        auto parts = split_csv_line(line);
        if (static_cast<int>(parts.size()) <= skip_cols) continue;
        Tuple t;
        //row_id = row_id + 1;
        t.id = row_id++;
        cout << t.id+1 << " ";
        for (int i = skip_cols; i < static_cast<int>(parts.size()); ++i)
        {
            t.x.push_back(stod(parts[i]));
            cout << parts[i] << " ";
        }
        ds.rows.push_back(std::move(t));
        //cout << "\n";
    }
    cout << "\n";
    return ds;
}

// ------------------------------------------------------------
// Subspace Construction
// ------------------------------------------------------------

void choose_rec(int start, int need, vector<int>& cur, int d, vector<Subspace>& out)
{
    if (need == 0)
    {
        out.push_back({cur});
        return;
    }
    for (int i = start; i <= d - need; ++i)
    {
        cur.push_back(i);
        
        //cout << "i=" << i << " d=" << d << " need=" << need << "\n";
        
        choose_rec(i + 1, need - 1, cur, d, out);
        cur.pop_back();
    }
}

vector<Subspace> all_subspaces_of_size(int d, int k)
{
    vector<Subspace> out;
    vector<int> cur;
    choose_rec(0, k, cur, d, out);
    return out;
}

vector<Subspace> make_selected_family(int d, int max_subspaces)
{
    vector<Subspace> fam;

    //In a default scenario prefer all 2D, then 3D, then 4D until reaching max_subspaces.
    //Feel free to change the selection and construction order
    for (int k : {2, 3, 4})
    {
        if (k > d) continue;
        auto subs = all_subspaces_of_size(d, k);
        for (auto& s : subs)
        {
            if ((int)fam.size() >= max_subspaces) return fam;
            fam.push_back(s);
        }
    }
    return fam;
}

string subspace_to_string(const Subspace& S)
{
    ostringstream out;
    out << "{";
    for (size_t i = 0; i < S.attrs.size(); ++i)
    {
        if (i) out << ", ";
        out << "a" << (S.attrs[i] + 1);
    }
    out << "}";
    return out.str();
}

void print_selected_family(const vector<Subspace>& F)
{
    cout << "Selected subspaces F (|F| = " << F.size() << "):\n";
    for (size_t i = 0; i < F.size(); ++i)
    {
        cout << "  S" << (i + 1) << " = " << subspace_to_string(F[i]) << "\n";
    }
}



// ------------------------------------------------------------
// 2D Skyline using Sort+Scan
// ------------------------------------------------------------

vector<int> skyline_2d_sort_scan(const DataSet& ds, const Subspace& S)
{
    assert(S.attrs.size() == 2);
    int a = S.attrs[0], b = S.attrs[1];
    vector<int> idx(ds.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int i, int j)
    {
        if (ds.rows[i].x[a] != ds.rows[j].x[a]) return ds.rows[i].x[a] < ds.rows[j].x[a];
        return ds.rows[i].x[b] < ds.rows[j].x[b];
    });

    vector<int> sky;
    double best_b = numeric_limits<double>::infinity();
    double best_a_for_best_b = numeric_limits<double>::infinity();

    for (int id : idx)
    {
        const double va = ds.rows[id].x[a];
        const double vb = ds.rows[id].x[b];

        if (vb < best_b)
        {
            // A new record-low second coordinate is always a skyline point.
            sky.push_back(id);
            best_b = vb;
            best_a_for_best_b = va;
        }
        else if (vb == best_b && va == best_a_for_best_b)
        {
            // An exact duplicate in this 2D projection is not strictly
            // dominated, so every duplicate item must remain in the skyline.
            sky.push_back(id);
        }
    }
    return sky;
}

// ------------------------------------------------------------
// Simple main-memory R*-tree.
// This produces a bulk-loaded R*-tree hierarchy.
// ------------------------------------------------------------

struct MBR
{
    vector<double> lo, hi;
    int count = 0;
};

struct RNode
{
    MBR box;
    bool leaf = false;
    vector<int> tuple_ids;        // used if leaf
    vector<int> children;         // used if internal
};

struct RStarTree
{
    Subspace S;
    vector<RNode> nodes;
    int root = -1;
};

MBR make_mbr_for_tuple(const Tuple& t, const Subspace& S)
{
    MBR b;
    int m = (int)S.attrs.size();
    b.lo.resize(m);
    b.hi.resize(m);
    for (int i = 0; i < m; ++i)
    {
        b.lo[i] = b.hi[i] = t.x[S.attrs[i]];
    }
    b.count = 1;
    return b;
}

MBR merge_mbr(const MBR& A, const MBR& B)
{
    MBR C;
    int m = (int)A.lo.size();
    C.lo.resize(m);
    C.hi.resize(m);
    for (int i = 0; i < m; ++i)
    {
        C.lo[i] = min(A.lo[i], B.lo[i]);
        C.hi[i] = max(A.hi[i], B.hi[i]);
    }
    C.count = A.count + B.count;
    return C;
}

Tuple mbr_lower_left_as_tuple(const MBR& b)
{
    Tuple t;
    t.id = -1;
    t.x = b.lo;
    return t;
}

Tuple mbr_upper_right_as_tuple(const MBR& b)
{
    Tuple t;
    t.id = -1;
    t.x = b.hi;
    return t;
}

RStarTree build_bulk_tree(const DataSet& ds, const Subspace& S, int leaf_cap = 96, int fanout = 24)
{
    vector<int> ids(ds.size());
    iota(ids.begin(), ids.end(), 0);

    // Simple STR-style sort by first subspace axis for readability.
    int first = S.attrs[0];
    sort(ids.begin(), ids.end(), [&](int i, int j)
    {
        return ds.rows[i].x[first] < ds.rows[j].x[first];
    });

    RStarTree tree;
    tree.S = S;

    vector<int> current_level;

    // Build leaves.
    for (int i = 0; i < (int)ids.size(); i += leaf_cap)
    {
        RNode node;
        node.leaf = true;
        int end = min((int)ids.size(), i + leaf_cap);
        node.tuple_ids.assign(ids.begin() + i, ids.begin() + end);
        bool first_box = true;
        for (int tid : node.tuple_ids)
        {
            MBR tb = make_mbr_for_tuple(ds.rows[tid], S);
            if (first_box)
            {
                node.box = tb;
                first_box = false;
            }
            else
            {
                node.box = merge_mbr(node.box, tb);
            }
        }
        tree.nodes.push_back(std::move(node));
        current_level.push_back((int)tree.nodes.size() - 1);
    }

    // Build internal levels.
    while ((int)current_level.size() > 1)
    {
        vector<int> next_level;
        for (int i = 0; i < (int)current_level.size(); i += fanout)
        {
            RNode node;
            node.leaf = false;
            int end = min((int)current_level.size(), i + fanout);
            node.children.assign(current_level.begin() + i, current_level.begin() + end);
            bool first_box = true;
            for (int cid : node.children)
            {
                if (first_box)
                {
                    node.box = tree.nodes[cid].box;
                    first_box = false;
                }
                else
                {
                    node.box = merge_mbr(node.box, tree.nodes[cid].box);
                }
            }
            tree.nodes.push_back(std::move(node));
            next_level.push_back((int)tree.nodes.size() - 1);
        }
        current_level.swap(next_level);
    }

    tree.root = current_level.front();
    return tree;
}

// ------------------------------------------------------------
// Skyline via BBS-style best-first traversal
// ------------------------------------------------------------

bool point_dominates_box_lower_left(const Tuple& p, const MBR& b)
{
    bool strict = false;
    for (size_t i = 0; i < b.lo.size(); ++i)
    {
        if (p.x[i] > b.lo[i]) return false;
        if (p.x[i] < b.lo[i]) strict = true;
    }
    return strict;
}

bool dominates_projected(const Tuple& p_full, const Tuple& q_full, const Subspace& S)
{
    bool strict = false;
    for (int idx : S.attrs)
    {
        if (p_full.x[idx] > q_full.x[idx]) return false;
        if (p_full.x[idx] < q_full.x[idx]) strict = true;
    }
    return strict;
}



// Check whether a FULL tuple dominates a PROJECTED point represented as a vector
// in the coordinate order of subspace S.
bool dominates_full_vs_projected_coords(const Tuple& full_p, const vector<double>& proj_q, const Subspace& S)
{
    bool strict = false;
    for (size_t i = 0; i < S.attrs.size(); ++i)
    {
        double pv = full_p.x[S.attrs[i]];
        double qv = proj_q[i];
        if (pv > qv) return false;
        if (pv < qv) strict = true;
    }
    return strict;
}

// Check whether the lower-left corner of an MBR is dominated by the current skyline.
bool mbr_lower_left_dominated_by_current_skyline(const MBR& box, const vector<int>& sky, const DataSet& ds, const Subspace& S)
{
    for (int sid : sky)
    {
        if (dominates_full_vs_projected_coords(ds.rows[sid], box.lo, S)) return true;
    }
    return false;
}


bool tuple_dominated_by_current_skyline(const Tuple& p, const vector<int>& sky, const DataSet& ds, const Subspace& S)
{
    for (int sid : sky)
    {
        if (dominates_projected(ds.rows[sid], p, S)) return true;
    }
    return false;
}


bool dominated_by_current_skyline(const Tuple& p, const vector<int>& sky, const DataSet& ds, const Subspace& S)
{
    for (int sid : sky)
    {
        if (dominates_projected(ds.rows[sid], p, S)) return true;
    }
    return false;
}



vector<int> skyline_bbs_rstar(const DataSet& ds, const RStarTree& tree)
{
    struct Entry
    {
        double key;
        int node_id;
        bool operator<(const Entry& other) const { return key > other.key; } // min-heap behavior
    };

    priority_queue<Entry> pq;
    pq.push({accumulate(tree.nodes[tree.root].box.lo.begin(), tree.nodes[tree.root].box.lo.end(), 0.0), tree.root});

    vector<int> sky;
    const Subspace& S = tree.S;

    while (!pq.empty())
    {
        Entry e = pq.top();
        pq.pop();
        const RNode& node = tree.nodes[e.node_id];

        if (mbr_lower_left_dominated_by_current_skyline(node.box, sky, ds, S))
        {
            continue;
        }

        if (node.leaf)
        {
            for (int tid : node.tuple_ids)
            {
                if (!tuple_dominated_by_current_skyline(ds.rows[tid], sky, ds, S))
                {
                    sky.push_back(tid);
                }
            }
        }
        else
        {
            for (int cid : node.children)
            {
                const RNode& ch = tree.nodes[cid];
                if (!mbr_lower_left_dominated_by_current_skyline(ch.box, sky, ds, S))
                {
                    double key = accumulate(ch.box.lo.begin(), ch.box.lo.end(), 0.0);
                    pq.push({key, cid});
                }
            }
        }
    }

    // Final filtering to remove any false positives
    vector<int> filtered;
    for (int tid : sky)
    {
        bool dom = false;
        for (int other : sky)
        {
            if (other != tid && dominates_projected(ds.rows[other], ds.rows[tid], S))
            {
                dom = true;
                break;
            }
        }
        if (!dom) filtered.push_back(tid);
    }

    sort(filtered.begin(), filtered.end());
    filtered.erase(unique(filtered.begin(), filtered.end()), filtered.end());
    return filtered;
}




// ------------------------------------------------------------
// Influence Counting Functions
// ------------------------------------------------------------

long long exact_influence_count_naive(const DataSet& ds, int pid, const Subspace& S)
{
    long long cnt = 0;
    const Tuple& p = ds.rows[pid];
    for (const auto& q : ds.rows)
    {
        if (q.id == p.id) continue;
        if (dominates_projected(p, q, S)) ++cnt;
    }
    return cnt;
}


double exact_influence_naive(const DataSet& ds, int pid, const Subspace& S)
{
    const long long cnt = exact_influence_count_naive(ds, pid, S);
    return ds.size() <= 1
        ? 0.0
        : static_cast<double>(cnt) / static_cast<double>(ds.size() - 1);
}

bool point_dominates_box_upper_right_projected(const Tuple& p, const MBR& b, const Subspace& S)
{
    bool strict = false;
    for (size_t i = 0; i < b.hi.size(); ++i)
    {
        double pv = p.x[S.attrs[i]];
        if (pv > b.hi[i]) return false;
        if (pv < b.hi[i]) strict = true;
    }
    return strict;
}

bool point_dominates_box_lower_left_projected(const Tuple& p, const MBR& b, const Subspace& S)
{
    bool strict = false;
    for (size_t i = 0; i < b.lo.size(); ++i)
    {
        double pv = p.x[S.attrs[i]];
        if (pv > b.lo[i]) return false;
        if (pv < b.lo[i]) strict = true;
    }
    return strict;
}

bool node_can_contain_dominated_points(const Tuple& p, const MBR& b, const Subspace& S)
{
    // If p is worse than the node's lower-left in any coordinate, some points may still be dominated,
    // so this test is conservative. We only return false when domination is impossible.
    bool possible = true;
    bool can_be_strict = false;
    for (size_t i = 0; i < b.lo.size(); ++i)
    {
        double pv = p.x[S.attrs[i]];
        if (pv > b.hi[i]) return false; // p exceeds even the best coordinate in node -> impossible.
        if (pv < b.hi[i]) can_be_strict = true;
    }
    return possible && can_be_strict;
}


void visit_exact_influence_rstar(const DataSet& ds, int pid, const RStarTree& tree, int nid, long long& cnt)
{
    const Tuple& p = ds.rows[pid];
    const Subspace& S = tree.S;
    const RNode& node = tree.nodes[nid];

    if (point_dominates_box_lower_left_projected(p, node.box, S))
    {
        cnt += node.box.count;
        return;
    }

    if (!node_can_contain_dominated_points(p, node.box, S))
    {
        return;
    }

    if (node.leaf)
    {
        for (int i = 0; i < (int)node.tuple_ids.size(); ++i)
        {
            int tid = node.tuple_ids[i];

            if (tid == pid)
            {
                continue;
            }

            if (dominates_projected(p, ds.rows[tid], S))
            {
                ++cnt;
            }
        }
    }
    else
    {
        for (int i = 0; i < (int)node.children.size(); ++i)
        {
            int cid = node.children[i];
            visit_exact_influence_rstar(ds, pid, tree, cid, cnt);
        }
    }
}




long long exact_influence_count_rstar(const DataSet& ds, int pid, const RStarTree& tree)
{
    long long cnt = 0;
    visit_exact_influence_rstar(ds, pid, tree, tree.root, cnt);
    return cnt;
}


double exact_influence_rstar(const DataSet& ds, int pid, const RStarTree& tree)
{
    const long long cnt = exact_influence_count_rstar(ds, pid, tree);
    return ds.size() <= 1
        ? 0.0
        : static_cast<double>(cnt) / static_cast<double>(ds.size() - 1);
}





// ------------------------------------------------------------
// SCDS Algorithm
// ------------------------------------------------------------

struct SubspaceInfo
{
    Subspace S;
    vector<int> skyline_ids;
    unordered_set<int> skyline_set;
    optional<RStarTree> tree;   // for 3D+
};

struct ResultItem
{
    int tuple_id;
    double score;
};

vector<ResultItem> scds(const DataSet& ds, const vector<Subspace>& F, int k, double lambda)
{
    vector<ResultItem> finished;
    if (ds.size() == 0 || F.empty() || k <= 0)
    {
        return finished;
    }

    vector<SubspaceInfo> infos;
    infos.reserve(F.size());
    unordered_set<int> UF;

    // Compute subspace skylines and create the union candidate set.
    for (const auto& S : F)
    {
        SubspaceInfo info;
        info.S = S;
        if (static_cast<int>(S.attrs.size()) == 2)
        {
            info.skyline_ids = skyline_2d_sort_scan(ds, S);
        }
        else
        {
            info.tree = build_bulk_tree(ds, S);
            info.skyline_ids = skyline_bbs_rstar(ds, *info.tree);
        }
        for (int id : info.skyline_ids)
        {
            info.skyline_set.insert(id);
            UF.insert(id);
        }
        infos.push_back(std::move(info));
    }

    vector<int> candidates(UF.begin(), UF.end());
    cout << "Candidates |UF|= " << UF.size() << "\n";

    vector<int> order(infos.size());
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b)
    {
        return infos[a].skyline_ids.size() > infos[b].skyline_ids.size();
    });

    // Heap construction
    struct HeapItem
    {
        double score;
        int tuple_id;

        bool operator<(const HeapItem& other) const
        {
            if (score != other.score) return score > other.score;
            return tuple_id < other.tuple_id;
        }
    };

    priority_queue<HeapItem> heap;
    double threshold = -numeric_limits<double>::infinity();

    const long double f_count = static_cast<long double>(F.size());
    const long double influence_denominator =
        ds.size() <= 1
            ? 1.0L
            : f_count * static_cast<long double>(ds.size() - 1);

    cout << "Non-pruned candidates calculations:\n";
    
    
    // Candidates Calculations
    for (int pid : candidates)
    {
        long long consensus_count = 0;
        long long dominated_count_sum = 0;
        bool alive = true;

        for (int pos = 0; pos < static_cast<int>(order.size()); ++pos)
        {
            const int si = order[pos];
            const auto& info = infos[si];

            if (info.skyline_set.count(pid))
            {
                ++consensus_count;
            }

            long long dominated_count = 0;
            if (static_cast<int>(info.S.attrs.size()) == 2)
            {
                dominated_count = exact_influence_count_naive(ds, pid, info.S);
            }
            else
            {
                dominated_count = exact_influence_count_rstar(ds, pid, *info.tree);
            }
            dominated_count_sum += dominated_count;

            const long double consensus_partial = static_cast<long double>(consensus_count) / f_count;
            const long double influence_partial =
                ds.size() <= 1
                    ? 0.0L
                    : static_cast<long double>(dominated_count_sum) / influence_denominator;
            const double partial = static_cast<double>( static_cast<long double>(lambda) * consensus_partial + static_cast<long double>(1.0 - lambda) * influence_partial);

            // Every unprocessed subspace can contribute at most 1 / |F|.
            const int remaining_count = static_cast<int>(order.size()) - pos - 1;
            const double ub = partial + static_cast<double>( static_cast<long double>(remaining_count) / f_count);

            // Equality cannot be pruned because tuple ID breaks score ties.
            if (static_cast<int>(heap.size()) >= k && ub < threshold)
            {
                alive = false;
                break;
            }
        }

        if (alive)
        {
            const long double consensus = static_cast<long double>(consensus_count) / f_count;
            const long double influence =
                ds.size() <= 1
                    ? 0.0L
                    : static_cast<long double>(dominated_count_sum) / influence_denominator;
            const double final_score = static_cast<double>( static_cast<long double>(lambda) * consensus + static_cast<long double>(1.0 - lambda) * influence);

            heap.push({final_score, pid});

            cout << "candidate:" << pid + 1 << " final score:" << final_score << "(consF=" << static_cast<double>(consensus) << ",infF=" << static_cast<double>(influence) << ")\n";

            if (static_cast<int>(heap.size()) > k)
            {
                heap.pop();
            }
            if (static_cast<int>(heap.size()) == k)
            {
                threshold = heap.top().score;
            }
        }
    }

    while (!heap.empty())
    {
        finished.push_back({heap.top().tuple_id, heap.top().score});
        heap.pop();
    }

    sort(finished.begin(), finished.end(), [](const ResultItem& a, const ResultItem& b)
    {
        if (a.score != b.score) return a.score > b.score;
        return a.tuple_id < b.tuple_id;
    });

    return finished;
}




// ------------------------------------------------------------
// Baseline Algorithm (BL)
//
// This implementation intentionally uses only the original dataset and
// the selected subspace family F. It performs pair-wise comparisons
// for every skyline and exhaustive candidate-to-tuple comparisons
// for every influence value. It uses no R-tree, sort-and-scan
// skyline method, priority ordering, upper bound, pruning rules,
// or top-k heap.
// ------------------------------------------------------------

vector<int> skyline_bruteforce_pairwise(const DataSet& ds, const Subspace& S)
{
    vector<int> skyline;
    skyline.reserve(ds.size());

    for (int pid = 0; pid < static_cast<int>(ds.size()); ++pid)
    {
        bool is_dominated = false;
        for (int qid = 0; qid < static_cast<int>(ds.size()); ++qid)
        {
            if (qid == pid) continue;
            if (dominates_in_subspace(ds.rows[qid], ds.rows[pid], S))
            {
                is_dominated = true;
                break;
            }
        }

        if (!is_dominated)
        {
            skyline.push_back(pid);
        }
    }

    return skyline;
}


long long influence_count_bruteforce_pairwise(const DataSet& ds, int pid, const Subspace& S)
{
    long long dominated_count = 0;

    // Exhaustively compare candidate p with every other dataset tuple q.
    for (int qid = 0; qid < static_cast<int>(ds.size()); ++qid)
    {
        if (qid == pid) continue;
        if (dominates_in_subspace(ds.rows[pid], ds.rows[qid], S))
        {
            ++dominated_count;
        }
    }

    return dominated_count;
}


vector<ResultItem> brute_force_scds(const DataSet& ds, const vector<Subspace>& F, int k, double lambda)
{
    vector<ResultItem> results;

    if (ds.size() == 0 || F.empty() || k <= 0)
    {
        return results;
    }

    const int n = static_cast<int>(ds.size());
    const int f_count = static_cast<int>(F.size());

    // skyline_membership[si][pid] == 1 exactly when tuple pid belongs to the skyline of subspace F[si].
    vector<vector<unsigned char>> skyline_membership(f_count, vector<unsigned char>(n, 0));
    vector<unsigned char> candidate_flag(n, 0);

    // Phase 1: derive every subspace skyline by exhaustive pair comparison,
    // and derive U_F as the union of those skylines.
    for (int si = 0; si < f_count; ++si)
    {
        cout << "Compute Skyline on subspace:" << subspace_to_string(F[si]) << "\n";
        vector<int> skyline_ids = skyline_bruteforce_pairwise(ds, F[si]);
        for (int pid : skyline_ids)
        {
            skyline_membership[si][pid] = 1;
            candidate_flag[pid] = 1;
        }
    }

    int candidate_count = 0;
    for (unsigned char flag : candidate_flag)
    {
        if (flag) ++candidate_count;
    }
    cout << "Candidates |UF|= " << candidate_count << "\n";

    results.reserve(candidate_count);

    // Phase 2: score every candidate fully. There is no bound and no pruning.
    for (int pid = 0; pid < n; ++pid)
    {
        if (!candidate_flag[pid]) continue;

        long long consensus_count = 0;
        long long dominated_count_sum = 0;

        // Process subspaces exactly in the user-defined order in F.
        for (int si = 0; si < f_count; ++si)
        {
            if (skyline_membership[si][pid])
            {
                ++consensus_count;
            }
            dominated_count_sum += influence_count_bruteforce_pairwise(ds, pid, F[si]);
        }

        const long double consensus = static_cast<long double>(consensus_count) / static_cast<long double>(f_count);
        const long double influence =
            ds.size() <= 1
                ? 0.0L
                : static_cast<long double>(dominated_count_sum) / (static_cast<long double>(f_count) * static_cast<long double>(ds.size() - 1));
        const double score = static_cast<double>( static_cast<long double>(lambda) * consensus + static_cast<long double>(1.0 - lambda) * influence);

        cout << "candidate:" << pid + 1 << " final score:" << score << "(consF=" << static_cast<double>(consensus) << ",infF=" << static_cast<double>(influence) << ")\n";

        results.push_back({pid, score});
    }

    // Rank only after every candidate has been evaluated completely.
    sort(results.begin(), results.end(), [](const ResultItem& a, const ResultItem& b)
    {
        if (a.score != b.score) return a.score > b.score;
        return a.tuple_id < b.tuple_id;
    });

    if (static_cast<int>(results.size()) > k)
    {
        results.resize(k);
    }

    return results;
}

// ------------------------------------------------------------
// Output results in a file
// ------------------------------------------------------------

void append_csv_row(ofstream& fout, const vector<string>& cols)
{
    for (int i = 0; i < (int)cols.size(); ++i)
    {
        if (i) fout << ',';
        fout << cols[i];
    }
    fout << '\n';
}


// ------------------------------------------------------------
// Real-data run functions
// ------------------------------------------------------------

void run_real_case(const string& csv_path, double lambda, int k, const string& out_topk)
{
    DataSet ds = load_csv_numeric(csv_path, 1);
    int d = ds.dim();
    auto F = make_selected_family(d, 20); // set the desired number of subspaces here
    
    print_selected_family(F);
    
    auto ans = scds(ds, F, k, lambda);

    ofstream fout(out_topk);
    if (!fout)
    {
        throw runtime_error("Cannot create output CSV: " + out_topk);
    }
    
    append_csv_row(fout, {"tuple_id", "score"});
    for (const auto& r : ans)
    {
        append_csv_row(fout, {to_string(r.tuple_id+1), to_string(r.score)});  //we use id+1 as counting starts from zero
    }
    
}

void run_real_case_bruteforce(const string& csv_path, double lambda, int k, const string& out_topk)
{
    DataSet ds = load_csv_numeric(csv_path, 1);
    int d = ds.dim();
    auto F = make_selected_family(d, 20); // set the desired number of subspaces here

    print_selected_family(F);

    auto ans = brute_force_scds(ds, F, k, lambda);

    ofstream fout(out_topk);
    if (!fout)
    {
        throw runtime_error("Cannot create output CSV: " + out_topk);
    }

    append_csv_row(fout, {"tuple_id", "score"});
    for (const auto& r : ans)
    {
        append_csv_row(fout, {to_string(r.tuple_id + 1), to_string(r.score)});  //we use id+1 as counting starts from zero
    }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv)
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // RUN modes:
    //   1 = SCDS on synthetic data
    //   2 = SCDS on CSV data
    //   3 = BL on synthetic data
    //   4 = BL on CSV data
    
    int RUN = 1;
    

    if (RUN == 1)
    {
        int n = 100000;   // dataset cardinality (number of items)
        int d = 5;   // dataset dimensionality (number of attributes)
        double lambda = 0.5;   // balance parameter
        int Fsize = 12;   // number of selected subspaces
        string dist = "uniform";   // uniform | correlated | anticorrelated     (distribution)
        int k = 5;   // number of results

        DataSet ds = make_synthetic(n, d, dist, 42);
        auto F = make_selected_family(d, Fsize);

        print_selected_family(F);
        if (static_cast<int>(F.size()) < Fsize)
        {
            cout << "Warning: requested Fsize = " << Fsize << ", but only " << F.size() << " subspaces are available for d = " << d << "\n";
        }

        auto t0 = chrono::high_resolution_clock::now();
        auto ans = scds(ds, F, k, lambda);
        auto t1 = chrono::high_resolution_clock::now();
        double sec = chrono::duration<double>(t1 - t0).count();

        cout << "algorithm = SCDS\n";
        cout << "runtime_s = " << fixed << setprecision(6) << sec << "\n";
        cout << "topk_size = " << ans.size() << "\n";
        for (const auto& r : ans)
        {
            cout << r.tuple_id+1 << ',' << fixed << setprecision(8) << r.score << '\n';
        }
    }
    else if (RUN == 2)
    {
        string csv_path = "hotels.csv";    //put the full path here for the input file
        double lambda = 0.5;
        int k = 3;
        string out_path = "results.csv";    //put the full path here for the output file

        auto t0 = chrono::high_resolution_clock::now();
        run_real_case(csv_path, lambda, k, out_path);
        auto t1 = chrono::high_resolution_clock::now();
        double sec = chrono::duration<double>(t1 - t0).count();

        cout << "algorithm = SCDS\n";
        cout << "runtime_s = " << fixed << setprecision(6) << sec << "\n";
        cout << "Real-data run completed: " << out_path << "\n";
    }
    else if (RUN == 3)
    {
        int n = 100000;
        int d = 5;
        double lambda = 0.5;
        int Fsize = 12;
        string dist = "uniform";   // uniform | correlated | anticorrelated
        int k = 5;

        DataSet ds = make_synthetic(n, d, dist, 42);
        auto F = make_selected_family(d, Fsize);

        print_selected_family(F);
        if (static_cast<int>(F.size()) < Fsize)
        {
            cout << "Warning: requested Fsize = " << Fsize << ", but only " << F.size() << " subspaces are available for d = " << d << "\n";
        }

        auto t0 = chrono::high_resolution_clock::now();
        auto ans = brute_force_scds(ds, F, k, lambda);
        auto t1 = chrono::high_resolution_clock::now();
        double sec = chrono::duration<double>(t1 - t0).count();

        cout << "algorithm = BL\n";
        cout << "runtime_s = " << fixed << setprecision(6) << sec << "\n";
        cout << "topk_size = " << ans.size() << "\n";
        for (const auto& r : ans)
        {
            cout << r.tuple_id+1 << ',' << fixed << setprecision(8) << r.score << '\n';
        }
    }
    else if (RUN == 4)
    {
        string csv_path = "hotels.csv";
        double lambda = 0.5;
        int k = 3;
        string out_path = "results_bl.csv";

        auto t0 = chrono::high_resolution_clock::now();
        run_real_case_bruteforce(csv_path, lambda, k, out_path);
        auto t1 = chrono::high_resolution_clock::now();
        double sec = chrono::duration<double>(t1 - t0).count();

        cout << "algorithm = BL\n";
        cout << "runtime_s = " << fixed << setprecision(6) << sec << "\n";
        cout << "Real-data brute-force run completed: " << out_path << "\n";
    }
    else
    {
        cout << "Invalid RUN value " << RUN << ". Use 1, 2, 3, or 4.\n";
    }

    return 0;
}
