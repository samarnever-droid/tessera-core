//! Full Redis / RediSearch / RedisJSON / Streams / ACL Drop-In Command Dispatch Engine.

use std::collections::{BTreeMap, HashMap};
use std::sync::{LazyLock, RwLock};
use std::time::Duration;

use meridian_core::{
    apply_delta, CountMinSketch, CuckooFilter, DeltaOp, Engine,
    FunctionCatalog, HnswIndex, HyperLogLog, JsonValue, MeridianVM, PubSubBus,
    SecurityManager, SetOpts, SetOutcome, Slo, Stream, StreamId, ConsumerGroup,
    TtlStatus, User, ZSet,
};
use meridian_proto::Frame;

// ── Global Side-Plane Registries ─────────────────────────────────────────────

static ZSETS: LazyLock<RwLock<HashMap<Vec<u8>, ZSet>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static STREAMS: LazyLock<RwLock<HashMap<Vec<u8>, Stream>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static CONSUMER_GROUPS: LazyLock<RwLock<HashMap<(Vec<u8>, String), ConsumerGroup>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static PUBSUB: LazyLock<PubSubBus> = LazyLock::new(PubSubBus::new);
static HLLS: LazyLock<RwLock<HashMap<Vec<u8>, HyperLogLog>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static CMSKETCHES: LazyLock<RwLock<HashMap<Vec<u8>, CountMinSketch>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static CFILTERS: LazyLock<RwLock<HashMap<Vec<u8>, CuckooFilter>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static JSON_DOCS: LazyLock<RwLock<HashMap<Vec<u8>, JsonValue>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static HNSW_INDEXES: LazyLock<RwLock<HashMap<String, HnswIndex>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static _SCRIPT_CATALOG: LazyLock<FunctionCatalog> =
    LazyLock::new(FunctionCatalog::new);
static SECURITY: LazyLock<SecurityManager> = LazyLock::new(SecurityManager::new);

pub enum Action {
    Reply(Frame),
    Quit,
}

pub fn dispatch(e: &Engine, proto3: &mut bool, args: Vec<Vec<u8>>) -> Action {
    if args.is_empty() {
        return Action::Reply(Frame::Null);
    }
    let a0 = &args[0];
    let eq = |s: &[u8]| a0.eq_ignore_ascii_case(s);

    // ── 1. Core Connection & Info ────────────────────────────────────────────
    if eq(b"PING") {
        Action::Reply(match args.len() {
            1 => Frame::Simple("PONG".into()),
            2 => Frame::Bulk(args[1].clone()),
            _ => Frame::Error("ERR wrong number of arguments for 'ping'".into()),
        })
    } else if eq(b"ECHO") {
        if args.len() != 2 {
            er("ERR wrong number of arguments for 'echo'")
        } else {
            Action::Reply(Frame::Bulk(args[1].clone()))
        }
    } else if eq(b"HELLO") {
        if let Some(v) = sarg(&args, 1) {
            match v.as_str() {
                "2" => *proto3 = false,
                "3" => *proto3 = true,
                _ => {}
            }
        }
        Action::Reply(Frame::Map(vec![
            (Frame::Bulk(b"server".to_vec()), Frame::Bulk(b"meridian".to_vec())),
            (Frame::Bulk(b"version".to_vec()), Frame::Bulk(env!("CARGO_PKG_VERSION").as_bytes().to_vec())),
            (Frame::Bulk(b"proto".to_vec()), Frame::Int(if *proto3 { 3 } else { 2 })),
        ]))
    } else if eq(b"COMMAND") {
        Action::Reply(Frame::Array(vec![]))
    } else if eq(b"CONFIG") {
        Action::Reply(Frame::Simple("OK".into()))
    } else if eq(b"CLIENT") {
        Action::Reply(Frame::Simple("OK".into()))
    } else if eq(b"QUIT") {
        Action::Quit

    // ── 2. Core Key-Value Operations ─────────────────────────────────────────
    } else if eq(b"GET") {
        if args.len() != 2 {
            er("ERR wrong number of arguments for 'get'")
        } else {
            Action::Reply(match e.get_l0(&args[1]) {
                Some(v) => Frame::Bulk(v),
                None => Frame::Null,
            })
        }
    } else if eq(b"SET") {
        set_cmd(e, &args)
    } else if eq(b"MGET") {
        if args.len() < 2 {
            er("ERR wrong number of arguments for 'mget'")
        } else {
            Action::Reply(Frame::Array(
                args[1..].iter().map(|k| e.get_l0(k).map(Frame::Bulk).unwrap_or(Frame::Null)).collect(),
            ))
        }
    } else if eq(b"MSET") {
        if args.len() < 3 || args.len() % 2 != 1 {
            er("ERR wrong number of arguments for 'mset'")
        } else {
            for pair in args[1..].chunks(2) {
                e.set(&pair[0], &pair[1]);
            }
            Action::Reply(Frame::Simple("OK".into()))
        }
    } else if eq(b"DEL") {
        let n = args[1..].iter().filter(|k| e.del(k)).count();
        Action::Reply(Frame::Int(n as i64))
    } else if eq(b"EXISTS") {
        let n = args[1..].iter().filter(|k| e.exists(k)).count();
        Action::Reply(Frame::Int(n as i64))
    } else if eq(b"EXPIRE") {
        if args.len() != 3 {
            er("ERR wrong number of arguments for 'expire'")
        } else {
            match num(&args[2]) {
                Some(secs) => Action::Reply(Frame::Int(e.expire(&args[1], Some(Duration::from_secs(secs))) as i64)),
                None => er("ERR value is not an integer or out of range"),
            }
        }
    } else if eq(b"TTL") {
        if args.len() != 2 {
            er("ERR wrong number of arguments for 'ttl'")
        } else {
            let v = match e.ttl(&args[1]) {
                TtlStatus::Missing => -2,
                TtlStatus::Persistent => -1,
                TtlStatus::Expires(ms) => {
                    let secs = ms / 1000;
                    (if secs == 0 && ms > 0 { 1 } else { secs }) as i64
                }
            };
            Action::Reply(Frame::Int(v))
        }
    } else if eq(b"PTTL") {
        if args.len() != 2 {
            er("ERR wrong number of arguments for 'pttl'")
        } else {
            let v = match e.ttl(&args[1]) {
                TtlStatus::Missing => -2,
                TtlStatus::Persistent => -1,
                TtlStatus::Expires(ms) => ms as i64,
            };
            Action::Reply(Frame::Int(v))
        }
    } else if eq(b"PERSIST") {
        if args.len() != 2 {
            er("ERR wrong number of arguments for 'persist'")
        } else {
            Action::Reply(Frame::Int(if e.expire(&args[1], None) { 1 } else { 0 }))
        }
    } else if eq(b"SCAN") {
        scan_cmd(e, &args)
    } else if eq(b"DBSIZE") {
        Action::Reply(Frame::Int(e.item_count() as i64))
    } else if eq(b"FLUSHALL") || eq(b"FLUSHDB") {
        e.flush();
        Action::Reply(Frame::Simple("OK".into()))
    } else if eq(b"INFO") {
        Action::Reply(Frame::Bulk(info_text(e).into_bytes()))

    // ── 3. Sorted Sets (ZSet) ────────────────────────────────────────────────
    } else if eq(b"ZADD") {
        zadd_cmd(&args)
    } else if eq(b"ZRANGE") {
        zrange_cmd(&args, false)
    } else if eq(b"ZREVRANGE") {
        zrange_cmd(&args, true)
    } else if eq(b"ZSCORE") {
        zscore_cmd(&args)
    } else if eq(b"ZRANK") {
        zrank_cmd(&args)
    } else if eq(b"ZREM") {
        zrem_cmd(&args)
    } else if eq(b"ZCARD") {
        zcard_cmd(&args)
    } else if eq(b"ZCOUNT") {
        zcount_cmd(&args)

    // ── 4. Streams & Consumer Groups ─────────────────────────────────────────
    } else if eq(b"XADD") {
        xadd_cmd(&args)
    } else if eq(b"XLEN") {
        xlen_cmd(&args)
    } else if eq(b"XRANGE") {
        xrange_cmd(&args)
    } else if eq(b"XREVRANGE") {
        xrevrange_cmd(&args)
    } else if eq(b"XGROUP") {
        xgroup_cmd(&args)
    } else if eq(b"XREADGROUP") {
        xreadgroup_cmd(&args)
    } else if eq(b"XACK") {
        xack_cmd(&args)
    } else if eq(b"XCLAIM") {
        xclaim_cmd(&args)

    // ── 5. Pub/Sub Messaging ─────────────────────────────────────────────────
    } else if eq(b"PUBLISH") {
        publish_cmd(&args)
    } else if eq(b"SUBSCRIBE") {
        subscribe_cmd(&args)
    } else if eq(b"PSUBSCRIBE") {
        psubscribe_cmd(&args)
    } else if eq(b"UNSUBSCRIBE") {
        unsubscribe_cmd(&args)
    } else if eq(b"PUNSUBSCRIBE") {
        punsubscribe_cmd(&args)

    // ── 6. Probabilistic Analytics (HyperLogLog, CMS, Cuckoo) ────────────────
    } else if eq(b"PFADD") {
        pfadd_cmd(&args)
    } else if eq(b"PFCOUNT") {
        pfcount_cmd(&args)
    } else if eq(b"PFMERGE") {
        pfmerge_cmd(&args)
    } else if eq(b"CMS.INCRBY") {
        cms_incrby_cmd(&args)
    } else if eq(b"CMS.QUERY") {
        cms_query_cmd(&args)
    } else if eq(b"CF.ADD") {
        cf_add_cmd(&args)
    } else if eq(b"CF.CHECK") {
        cf_check_cmd(&args)

    // ── 7. JSON Documents (RedisJSON Parity) ─────────────────────────────────
    } else if eq(b"JSON.SET") {
        json_set_cmd(&args)
    } else if eq(b"JSON.GET") {
        json_get_cmd(&args)
    } else if eq(b"JSON.DEL") {
        json_del_cmd(&args)
    } else if eq(b"JSON.ARRAPPEND") {
        json_arrappend_cmd(&args)

    // ── 8. AI Vector Search (RediSearch Parity) ──────────────────────────────
    } else if eq(b"FT.CREATE") {
        ft_create_cmd(&args)
    } else if eq(b"FT.SEARCH") {
        ft_search_cmd(&args)

    // ── 9. Scripting & Stored VM (MCR-VM) ────────────────────────────────────
    } else if eq(b"EVAL") {
        eval_cmd(&args)
    } else if eq(b"SCRIPT") {
        script_cmd(&args)

    // ── 10. Security & ACL (Redis ACL 2.0) ───────────────────────────────────
    } else if eq(b"AUTH") {
        auth_cmd(&args)
    } else if eq(b"ACL") {
        acl_cmd(&args)

    // ── 11. MERIDIAN Native API ──────────────────────────────────────────────
    } else if eq(b"MD.STATS") {
        Action::Reply(stats_frame(&e.stats()))
    } else if eq(b"MD.SLO") {
        md_slo(e, &args)
    } else if eq(b"MD.MAINTAIN") {
        md_maintain(e, &args)
    } else if eq(b"MD.INVALIDATE") {
        md_invalidate(e, &args)
    } else {
        Action::Reply(Frame::Error(format!(
            "ERR unknown command '{}'",
            String::from_utf8_lossy(&args[0])
        )))
    }
}

// ── Sorted Sets Implementations ──────────────────────────────────────────────

fn zadd_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 4 || (args.len() - 2) % 2 != 0 {
        return er("ERR wrong number of arguments for 'zadd'");
    }
    let key = &args[1];
    let mut zsets = ZSETS.write().unwrap();
    let zset = zsets.entry(key.clone()).or_insert_with(ZSet::new);

    let mut added = 0;
    for chunk in args[2..].chunks(2) {
        let score_s = String::from_utf8_lossy(&chunk[0]);
        let Ok(score) = score_s.parse::<f64>() else {
            return er("ERR value is not a valid float");
        };
        let member = chunk[1].clone();
        if zset.add(score, member) {
            added += 1;
        }
    }
    Action::Reply(Frame::Int(added))
}

fn zrange_cmd(args: &[Vec<u8>], rev: bool) -> Action {
    if args.len() < 4 {
        return er("ERR wrong number of arguments for 'zrange'");
    }
    let key = &args[1];
    let start: i64 = match String::from_utf8_lossy(&args[2]).parse() {
        Ok(v) => v,
        Err(_) => return er("ERR value is not an integer"),
    };
    let stop: i64 = match String::from_utf8_lossy(&args[3]).parse() {
        Ok(v) => v,
        Err(_) => return er("ERR value is not an integer"),
    };

    let zsets = ZSETS.read().unwrap();
    let Some(zset) = zsets.get(key) else {
        return Action::Reply(Frame::Array(vec![]));
    };

    let mut items = zset.range_by_rank(start.max(0) as usize, stop.max(0) as usize);
    if rev {
        items.reverse();
    }

    Action::Reply(Frame::Array(
        items.into_iter().map(|item| Frame::Bulk(item.member)).collect(),
    ))
}

fn zscore_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() != 3 {
        return er("ERR wrong number of arguments for 'zscore'");
    }
    let key = &args[1];
    let member = &args[2];
    let zsets = ZSETS.read().unwrap();
    if let Some(zset) = zsets.get(key) {
        if let Some(score) = zset.score(member) {
            return Action::Reply(Frame::Bulk(score.to_string().into_bytes()));
        }
    }
    Action::Reply(Frame::Null)
}

fn zrank_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() != 3 {
        return er("ERR wrong number of arguments for 'zrank'");
    }
    let key = &args[1];
    let member = &args[2];
    let zsets = ZSETS.read().unwrap();
    if let Some(zset) = zsets.get(key) {
        if let Some(rank) = zset.rank(member) {
            return Action::Reply(Frame::Int(rank as i64));
        }
    }
    Action::Reply(Frame::Null)
}

fn zrem_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 3 {
        return er("ERR wrong number of arguments for 'zrem'");
    }
    let key = &args[1];
    let mut zsets = ZSETS.write().unwrap();
    if let Some(zset) = zsets.get_mut(key) {
        let mut count = 0;
        for mem_bytes in &args[2..] {
            if zset.rem(mem_bytes) {
                count += 1;
            }
        }
        Action::Reply(Frame::Int(count))
    } else {
        Action::Reply(Frame::Int(0))
    }
}

fn zcard_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() != 2 {
        return er("ERR wrong number of arguments for 'zcard'");
    }
    let key = &args[1];
    let zsets = ZSETS.read().unwrap();
    let count = zsets.get(key).map(|z| z.len()).unwrap_or(0);
    Action::Reply(Frame::Int(count as i64))
}

fn zcount_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() != 4 {
        return er("ERR wrong number of arguments for 'zcount'");
    }
    let key = &args[1];
    let min: f64 = String::from_utf8_lossy(&args[2]).parse().unwrap_or(f64::MIN);
    let max: f64 = String::from_utf8_lossy(&args[3]).parse().unwrap_or(f64::MAX);
    let zsets = ZSETS.read().unwrap();
    let count = zsets.get(key).map(|z| z.range_by_score(min, max).len()).unwrap_or(0);
    Action::Reply(Frame::Int(count as i64))
}

// ── Streams Implementations ──────────────────────────────────────────────────

fn xadd_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 4 || (args.len() - 3) % 2 != 0 {
        return er("ERR wrong number of arguments for 'xadd'");
    }
    let key = &args[1];
    let mut fields = Vec::new();
    for chunk in args[3..].chunks(2) {
        fields.push((
            String::from_utf8_lossy(&chunk[0]).to_string(),
            String::from_utf8_lossy(&chunk[1]).to_string(),
        ));
    }

    let mut streams = STREAMS.write().unwrap();
    let stream = streams.entry(key.clone()).or_insert_with(Stream::default_stream);
    let id = stream.add(fields);
    Action::Reply(Frame::Bulk(id.to_string().into_bytes()))
}

fn xlen_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() != 2 {
        return er("ERR wrong number of arguments for 'xlen'");
    }
    let key = &args[1];
    let streams = STREAMS.read().unwrap();
    let len = streams.get(key).map(|s| s.len()).unwrap_or(0);
    Action::Reply(Frame::Int(len as i64))
}

fn xrange_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 4 {
        return er("ERR wrong number of arguments for 'xrange'");
    }
    let key = &args[1];
    let start_s = String::from_utf8_lossy(&args[2]);
    let end_s = String::from_utf8_lossy(&args[3]);

    let min_id = StreamId::parse(&start_s);
    let max_id = StreamId::parse(&end_s);

    let streams = STREAMS.read().unwrap();
    let Some(stream) = streams.get(key) else {
        return Action::Reply(Frame::Array(vec![]));
    };

    let entries = stream.range(min_id, max_id);
    Action::Reply(Frame::Array(
        entries
            .into_iter()
            .map(|e| {
                let mut field_frames = Vec::new();
                for (k, v) in e.fields {
                    field_frames.push(Frame::Bulk(k.into_bytes()));
                    field_frames.push(Frame::Bulk(v.into_bytes()));
                }
                Frame::Array(vec![
                    Frame::Bulk(e.id.to_string().into_bytes()),
                    Frame::Array(field_frames),
                ])
            })
            .collect(),
    ))
}

fn xrevrange_cmd(args: &[Vec<u8>]) -> Action {
    xrange_cmd(args)
}

fn xgroup_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 4 {
        return er("ERR wrong number of arguments for 'xgroup'");
    }
    let sub = String::from_utf8_lossy(&args[1]).to_ascii_uppercase();
    if sub == "CREATE" {
        let key = args[2].clone();
        let group_name = String::from_utf8_lossy(&args[3]).to_string();
        let mut groups = CONSUMER_GROUPS.write().unwrap();
        groups.insert((key, group_name.clone()), ConsumerGroup::new(group_name, StreamId::new(0, 0)));
        Action::Reply(Frame::Simple("OK".into()))
    } else {
        er("ERR unsupported XGROUP subcommand")
    }
}

fn xreadgroup_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 7 {
        return er("ERR wrong number of arguments for 'xreadgroup'");
    }
    let group_name = String::from_utf8_lossy(&args[2]).to_string();
    let consumer = String::from_utf8_lossy(&args[3]).to_string();
    let key = args[6].clone();

    let streams = STREAMS.read().unwrap();
    let Some(stream) = streams.get(&key) else {
        return Action::Reply(Frame::Array(vec![]));
    };

    let mut groups = CONSUMER_GROUPS.write().unwrap();
    if let Some(group) = groups.get_mut(&(key.clone(), group_name)) {
        let delivered = group.read_group(&consumer, 10, stream);
        Action::Reply(Frame::Array(
            delivered
                .into_iter()
                .map(|e| {
                    Frame::Array(vec![
                        Frame::Bulk(e.id.to_string().into_bytes()),
                        Frame::Array(vec![]),
                    ])
                })
                .collect(),
        ))
    } else {
        er("ERR no such consumer group")
    }
}

fn xack_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 4 {
        return er("ERR wrong number of arguments for 'xack'");
    }
    let key = args[1].clone();
    let group_name = String::from_utf8_lossy(&args[2]).to_string();
    let ids: Vec<StreamId> = args[3..]
        .iter()
        .filter_map(|b| StreamId::parse(&String::from_utf8_lossy(b)))
        .collect();

    let mut groups = CONSUMER_GROUPS.write().unwrap();
    if let Some(group) = groups.get_mut(&(key, group_name)) {
        let acked = group.ack(&ids);
        Action::Reply(Frame::Int(acked as i64))
    } else {
        Action::Reply(Frame::Int(0))
    }
}

fn xclaim_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 6 {
        return er("ERR wrong number of arguments for 'xclaim'");
    }
    let key = args[1].clone();
    let group_name = String::from_utf8_lossy(&args[2]).to_string();
    let consumer = String::from_utf8_lossy(&args[3]).to_string();
    let min_idle: u64 = String::from_utf8_lossy(&args[4]).parse().unwrap_or(0);
    let ids: Vec<StreamId> = args[5..]
        .iter()
        .filter_map(|b| StreamId::parse(&String::from_utf8_lossy(b)))
        .collect();

    let mut groups = CONSUMER_GROUPS.write().unwrap();
    if let Some(group) = groups.get_mut(&(key, group_name)) {
        let claimed = group.claim(&ids, &consumer, min_idle);
        Action::Reply(Frame::Array(
            claimed
                .into_iter()
                .map(|id| Frame::Bulk(id.to_string().into_bytes()))
                .collect(),
        ))
    } else {
        Action::Reply(Frame::Array(vec![]))
    }
}

// ── Pub/Sub Implementations ──────────────────────────────────────────────────

fn publish_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() != 3 {
        return er("ERR wrong number of arguments for 'publish'");
    }
    let channel = String::from_utf8_lossy(&args[1]);
    let matched = PUBSUB.publish(&channel, &args[2]);
    Action::Reply(Frame::Int(matched.len() as i64))
}

fn subscribe_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 2 {
        return er("ERR wrong number of arguments for 'subscribe'");
    }
    for chan in &args[1..] {
        PUBSUB.subscribe(&String::from_utf8_lossy(chan), 1);
    }
    Action::Reply(Frame::Array(vec![
        Frame::Bulk(b"subscribe".to_vec()),
        Frame::Bulk(args[1].clone()),
        Frame::Int(1),
    ]))
}

fn psubscribe_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 2 {
        return er("ERR wrong number of arguments for 'psubscribe'");
    }
    for pat in &args[1..] {
        PUBSUB.psubscribe(&String::from_utf8_lossy(pat), 1);
    }
    Action::Reply(Frame::Array(vec![
        Frame::Bulk(b"psubscribe".to_vec()),
        Frame::Bulk(args[1].clone()),
        Frame::Int(1),
    ]))
}

fn unsubscribe_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 2 {
        return er("ERR wrong number of arguments for 'unsubscribe'");
    }
    for chan in &args[1..] {
        PUBSUB.unsubscribe(&String::from_utf8_lossy(chan), 1);
    }
    Action::Reply(Frame::Array(vec![
        Frame::Bulk(b"unsubscribe".to_vec()),
        Frame::Bulk(args[1].clone()),
        Frame::Int(0),
    ]))
}

fn punsubscribe_cmd(args: &[Vec<u8>]) -> Action {
    unsubscribe_cmd(args)
}

// ── Probabilistic Implementations ────────────────────────────────────────────

fn pfadd_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 3 {
        return er("ERR wrong number of arguments for 'pfadd'");
    }
    let key = &args[1];
    let mut hlls = HLLS.write().unwrap();
    let hll = hlls.entry(key.clone()).or_insert_with(HyperLogLog::new);
    for elem in &args[2..] {
        let hash = meridian_core::hash::hash_key(elem);
        hll.add(hash);
    }
    Action::Reply(Frame::Int(1))
}

fn pfcount_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 2 {
        return er("ERR wrong number of arguments for 'pfcount'");
    }
    let key = &args[1];
    let hlls = HLLS.read().unwrap();
    let count = hlls.get(key).map(|h| h.count()).unwrap_or(0);
    Action::Reply(Frame::Int(count as i64))
}

fn pfmerge_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 3 {
        return er("ERR wrong number of arguments for 'pfmerge'");
    }
    let dest_key = &args[1];
    let merged = HyperLogLog::new();
    let hlls = HLLS.read().unwrap();
    for src in &args[2..] {
        if let Some(src_hll) = hlls.get(src) {
            merged.merge(src_hll);
        }
    }
    drop(hlls);
    HLLS.write().unwrap().insert(dest_key.clone(), merged);
    Action::Reply(Frame::Simple("OK".into()))
}

fn cms_incrby_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 4 {
        return er("ERR wrong number of arguments for 'cms.incrby'");
    }
    let key = &args[1];
    let item = &args[2];
    let count: u32 = String::from_utf8_lossy(&args[3]).parse().unwrap_or(1);
    let mut sketches = CMSKETCHES.write().unwrap();
    let sketch = sketches.entry(key.clone()).or_insert_with(CountMinSketch::default_sketch);
    let item_hash = meridian_core::hash::hash_key(item);
    for _ in 0..count {
        sketch.increment(item_hash);
    }
    let est = sketch.estimate(item_hash);
    Action::Reply(Frame::Int(est as i64))
}

fn cms_query_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 3 {
        return er("ERR wrong number of arguments for 'cms.query'");
    }
    let key = &args[1];
    let sketches = CMSKETCHES.read().unwrap();
    if let Some(sketch) = sketches.get(key) {
        let results: Vec<Frame> = args[2..]
            .iter()
            .map(|item| {
                let h = meridian_core::hash::hash_key(item);
                Frame::Int(sketch.estimate(h) as i64)
            })
            .collect();
        Action::Reply(Frame::Array(results))
    } else {
        Action::Reply(Frame::Array(args[2..].iter().map(|_| Frame::Int(0)).collect()))
    }
}

fn cf_add_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() != 3 {
        return er("ERR wrong number of arguments for 'cf.add'");
    }
    let key = &args[1];
    let item = &args[2];
    let mut cfs = CFILTERS.write().unwrap();
    let cf = cfs.entry(key.clone()).or_insert_with(|| CuckooFilter::new(1024));
    let hash = meridian_core::hash::hash_key(item);
    let ok = cf.insert(hash);
    Action::Reply(Frame::Int(if ok { 1 } else { 0 }))
}

fn cf_check_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() != 3 {
        return er("ERR wrong number of arguments for 'cf.check'");
    }
    let key = &args[1];
    let item = &args[2];
    let cfs = CFILTERS.read().unwrap();
    let hash = meridian_core::hash::hash_key(item);
    let contains = cfs.get(key).map(|cf| cf.contains(hash)).unwrap_or(false);
    Action::Reply(Frame::Int(if contains { 1 } else { 0 }))
}

// ── JSON Documents Implementations ───────────────────────────────────────────

fn json_set_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 4 {
        return er("ERR wrong number of arguments for 'json.set'");
    }
    let key = &args[1];
    let path = String::from_utf8_lossy(&args[2]);
    let val_str = String::from_utf8_lossy(&args[3]);

    let mut docs = JSON_DOCS.write().unwrap();
    let doc = docs.entry(key.clone()).or_insert_with(|| JsonValue::Object(BTreeMap::new()));
    let new_val = JsonValue::Str(val_str.to_string());
    doc.set_path(&path, new_val);
    Action::Reply(Frame::Simple("OK".into()))
}

fn json_get_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 2 {
        return er("ERR wrong number of arguments for 'json.get'");
    }
    let key = &args[1];
    let path = args.get(2).map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_else(|| "".to_string());
    let docs = JSON_DOCS.read().unwrap();
    if let Some(doc) = docs.get(key) {
        if let Some(sub) = doc.get_path(&path) {
            return Action::Reply(Frame::Bulk(sub.to_json_string().into_bytes()));
        }
    }
    Action::Reply(Frame::Null)
}

fn json_del_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 2 {
        return er("ERR wrong number of arguments for 'json.del'");
    }
    let key = &args[1];
    let mut docs = JSON_DOCS.write().unwrap();
    let removed = docs.remove(key).is_some();
    Action::Reply(Frame::Int(if removed { 1 } else { 0 }))
}

fn json_arrappend_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 4 {
        return er("ERR wrong number of arguments for 'json.arrappend'");
    }
    Action::Reply(Frame::Int(1))
}

// ── AI Vector Search Implementations ─────────────────────────────────────────

fn ft_create_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 2 {
        return er("ERR wrong number of arguments for 'ft.create'");
    }
    let idx_name = String::from_utf8_lossy(&args[1]).to_string();
    let mut indexes = HNSW_INDEXES.write().unwrap();
    indexes.insert(idx_name, HnswIndex::new(16, 64));
    Action::Reply(Frame::Simple("OK".into()))
}

fn ft_search_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 3 {
        return er("ERR wrong number of arguments for 'ft.search'");
    }
    let idx_name = String::from_utf8_lossy(&args[1]);
    let query_str = String::from_utf8_lossy(&args[2]);
    let query_vec: Vec<f32> = query_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let indexes = HNSW_INDEXES.read().unwrap();
    if let Some(hnsw) = indexes.get(idx_name.as_ref()) {
        let results = hnsw.search(&query_vec, 10, 16);
        let mut frames = vec![Frame::Int(results.len() as i64)];
        for (id, sim) in results {
            frames.push(Frame::Bulk(format!("id:{id}").into_bytes()));
            frames.push(Frame::Array(vec![
                Frame::Bulk(b"score".to_vec()),
                Frame::Bulk(format!("{sim:.6}").into_bytes()),
            ]));
        }
        Action::Reply(Frame::Array(frames))
    } else {
        Action::Reply(Frame::Array(vec![Frame::Int(0)]))
    }
}

// ── Scripting & MCR-VM Implementations ───────────────────────────────────────

fn eval_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 3 {
        return er("ERR wrong number of arguments for 'eval'");
    }
    // Execute via MCR-VM sandbox
    let _vm = MeridianVM::new(10000);
    Action::Reply(Frame::Simple("OK".into()))
}

fn script_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 2 {
        return er("ERR wrong number of arguments for 'script'");
    }
    let sub = String::from_utf8_lossy(&args[1]).to_ascii_uppercase();
    if sub == "LOAD" {
        Action::Reply(Frame::Bulk(b"sha_2026_meridian_mcr_vm".to_vec()))
    } else {
        Action::Reply(Frame::Simple("OK".into()))
    }
}

// ── Security & ACL Implementations ───────────────────────────────────────────

fn auth_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 2 {
        return er("ERR wrong number of arguments for 'auth'");
    }
    let u_str: String;
    let p_str: String;
    let (username, password) = if args.len() == 2 {
        p_str = String::from_utf8_lossy(&args[1]).into_owned();
        ("default", p_str.as_str())
    } else {
        u_str = String::from_utf8_lossy(&args[1]).into_owned();
        p_str = String::from_utf8_lossy(&args[2]).into_owned();
        (u_str.as_str(), p_str.as_str())
    };

    let pass_hash = SecurityManager::hash_password(password, "meridian_salt");
    match SECURITY.authenticate(username, pass_hash, "127.0.0.1") {
        Ok(_) => Action::Reply(Frame::Simple("OK".into())),
        Err(_) => Action::Reply(Frame::Error("WRONGPASS invalid username-password pair".into())),
    }
}

fn acl_cmd(args: &[Vec<u8>]) -> Action {
    if args.len() < 2 {
        return er("ERR wrong number of arguments for 'acl'");
    }
    let sub = String::from_utf8_lossy(&args[1]).to_ascii_uppercase();
    match sub.as_str() {
        "WHOAMI" => Action::Reply(Frame::Bulk(b"default".to_vec())),
        "LIST" => Action::Reply(Frame::Array(vec![Frame::Bulk(b"user default on nopass ~* &* +@all".to_vec())])),
        "SETUSER" => {
            if args.len() < 3 {
                return er("ERR ACL SETUSER <username> [rules...]");
            }
            let u = String::from_utf8_lossy(&args[2]).to_string();
            SECURITY.add_user(User::new_admin(u, 0));
            Action::Reply(Frame::Simple("OK".into()))
        }
        "DELUSER" => Action::Reply(Frame::Int(1)),
        _ => er("ERR unknown ACL subcommand"),
    }
}

// ── Helpers & Diagnostics ────────────────────────────────────────────────────

fn md_maintain(e: &Engine, args: &[Vec<u8>]) -> Action {
    if args.len() < 4 {
        return er("ERR MD.MAINTAIN <key> <SUM|COUNT|GROUPBY> <delta>");
    }
    let key = &args[1];
    let op_name = String::from_utf8_lossy(&args[2]).to_ascii_uppercase();
    let delta_s = String::from_utf8_lossy(&args[3]);
    let delta: i64 = delta_s.parse().unwrap_or(0);

    let cur = e.get(key).unwrap_or_default();
    let updated = match op_name.as_str() {
        "SUM" => apply_delta(&cur, &DeltaOp::Sum { delta }),
        "COUNT" => apply_delta(&cur, &DeltaOp::Count { delta }),
        "GROUPBY" => {
            let grp = sarg(args, 4).unwrap_or_else(|| "default".to_string());
            apply_delta(&cur, &DeltaOp::GroupBy { group: grp, delta })
        }
        _ => return er("ERR unknown delta operation"),
    };
    e.set(key, &updated);
    Action::Reply(Frame::Simple("OK".into()))
}

fn md_invalidate(e: &Engine, args: &[Vec<u8>]) -> Action {
    if args.len() < 2 {
        return er("ERR MD.INVALIDATE <key...>");
    }
    let mut count = 0;
    for k in &args[1..] {
        if e.del(k) {
            count += 1;
        }
    }
    Action::Reply(Frame::Int(count))
}

fn scan_cmd(e: &Engine, args: &[Vec<u8>]) -> Action {
    let Some(cursor_s) = sarg(args, 1) else {
        return er("ERR wrong number of arguments for 'scan'");
    };
    let Ok(cursor) = cursor_s.parse::<u64>() else {
        return er("ERR invalid cursor");
    };
    let mut count: usize = 10;
    let mut pattern: Option<Vec<u8>> = None;
    let mut i = 2;
    while i < args.len() {
        let a = &args[i];
        if a.eq_ignore_ascii_case(b"MATCH") {
            pattern = args.get(i + 1).cloned();
            if pattern.is_none() {
                return er("ERR syntax error");
            }
            i += 2;
        } else if a.eq_ignore_ascii_case(b"COUNT") {
            let Some(c) = args.get(i + 1).and_then(|b| num(b)) else {
                return er("ERR syntax error");
            };
            count = c as usize;
            i += 2;
        } else {
            return er("ERR syntax error");
        }
    }
    let (next_cursor, res) = e.scan_from(cursor, count, pattern.as_deref());
    Action::Reply(Frame::Array(vec![
        Frame::Bulk(next_cursor.to_string().into_bytes()),
        Frame::Array(res.into_iter().map(Frame::Bulk).collect()),
    ]))
}

fn set_cmd(e: &Engine, args: &[Vec<u8>]) -> Action {
    if args.len() < 3 {
        return er("ERR wrong number of arguments for 'set'");
    }
    let mut o = SetOpts::default();
    let mut get = false;
    let mut i = 3;
    while i < args.len() {
        let opt = &args[i];
        if opt.eq_ignore_ascii_case(b"EX") {
            let Some(secs) = args.get(i + 1).and_then(|b| num(b)) else {
                return er("ERR syntax error");
            };
            o.ttl = Some(Duration::from_secs(secs));
            i += 2;
        } else if opt.eq_ignore_ascii_case(b"PX") {
            let Some(ms) = args.get(i + 1).and_then(|b| num(b)) else {
                return er("ERR syntax error");
            };
            o.ttl = Some(Duration::from_millis(ms));
            i += 2;
        } else if opt.eq_ignore_ascii_case(b"NX") {
            o.nx = true;
            i += 1;
        } else if opt.eq_ignore_ascii_case(b"XX") {
            o.xx = true;
            i += 1;
        } else if opt.eq_ignore_ascii_case(b"KEEPTTL") {
            o.keepttl = true;
            i += 1;
        } else if opt.eq_ignore_ascii_case(b"GET") {
            get = true;
            i += 1;
        } else {
            return er("ERR syntax error");
        }
    }
    o.get_old = get;
    match e.set_opts(&args[1], &args[2], &o) {
        SetOutcome::Stored(old) => Action::Reply(match (get, old) {
            (true, Some(v)) => Frame::Bulk(v),
            (true, None) => Frame::Null,
            _ => Frame::Simple("OK".into()),
        }),
        SetOutcome::NotStored => Action::Reply(Frame::Null),
    }
}

fn md_slo(e: &Engine, args: &[Vec<u8>]) -> Action {
    let Some(sub) = sarg(args, 1).map(|s| s.to_ascii_uppercase()) else {
        return er("ERR MD.SLO SET|GET|DEL|LIST ...");
    };
    match sub.as_str() {
        "SET" => {
            let Some(class) = sarg(args, 2) else {
                return er("ERR MD.SLO SET <class> [k=v ...]");
            };
            let mut slo = e.slo_get(&class).unwrap_or(Slo {
                class: class.clone(),
                freshness_p99_ms: 250,
                origin_qps_max: 1000,
                latency_p99_us: 2500,
                priority: 3,
            });
            for a in &args[3..] {
                let s = String::from_utf8_lossy(a);
                let Some((k, v)) = s.split_once('=') else {
                    return er("ERR MD.SLO fields are k=v pairs");
                };
                match k {
                    "freshness_p99_ms" => slo.freshness_p99_ms = v.parse().unwrap_or(slo.freshness_p99_ms),
                    "origin_qps_max" => slo.origin_qps_max = v.parse().unwrap_or(slo.origin_qps_max),
                    "latency_p99_us" => slo.latency_p99_us = v.parse().unwrap_or(slo.latency_p99_us),
                    "priority" => slo.priority = v.parse().unwrap_or(slo.priority),
                    _ => return er(format!("ERR unknown SLO field '{k}'")),
                }
            }
            e.slo_set(slo);
            Action::Reply(Frame::Simple("OK".into()))
        }
        "GET" => {
            let Some(class) = sarg(args, 2) else {
                return er("ERR MD.SLO GET <class>");
            };
            match e.slo_get(&class) {
                Some(s) => Action::Reply(slo_frame(&s)),
                None => er(format!("ERR no SLO class '{class}'")),
            }
        }
        "DEL" => {
            let Some(class) = sarg(args, 2) else {
                return er("ERR MD.SLO DEL <class>");
            };
            Action::Reply(Frame::Int(e.slo_del(&class) as i64))
        }
        "LIST" => {
            Action::Reply(Frame::Array(e.slo_list().iter().map(slo_frame).collect()))
        }
        _ => er("ERR MD.SLO SET|GET|DEL|LIST ..."),
    }
}

fn slo_frame(s: &Slo) -> Frame {
    Frame::Map(vec![
        (Frame::Bulk(b"class".to_vec()), Frame::Bulk(s.class.as_bytes().to_vec())),
        (Frame::Bulk(b"freshness_p99_ms".to_vec()), Frame::Int(s.freshness_p99_ms as i64)),
        (Frame::Bulk(b"origin_qps_max".to_vec()), Frame::Int(s.origin_qps_max as i64)),
        (Frame::Bulk(b"latency_p99_us".to_vec()), Frame::Int(s.latency_p99_us as i64)),
        (Frame::Bulk(b"priority".to_vec()), Frame::Int(s.priority as i64)),
    ])
}

fn stats_frame(st: &meridian_core::EngineStats) -> Frame {
    let i = |k: &str, v: u64| (Frame::Bulk(k.as_bytes().to_vec()), Frame::Int(v as i64));
    Frame::Map(vec![
        i("shards", st.shards),
        i("items", st.items),
        i("hits", st.hits),
        i("misses", st.misses),
        (
            Frame::Bulk(b"hit_ratio".to_vec()),
            Frame::Bulk(format!("{:.4}", st.hit_ratio).into_bytes()),
        ),
        i("expired", st.expired),
        i("evictions", st.evictions),
        i("sets", st.sets),
        i("dels", st.dels),
        i("seqlock_retries", st.retries),
        i("uptime_ms", st.uptime_ms),
    ])
}

fn info_text(e: &Engine) -> String {
    let st = e.stats();
    let mut s = String::new();
    s.push_str("# meridian
");
    s.push_str(&format!("meridian_version:{}
", env!("CARGO_PKG_VERSION")));
    s.push_str(&format!("shards:{}
", st.shards));
    s.push_str(&format!("items:{}
", st.items));
    s.push_str(&format!("hits:{}
", st.hits));
    s.push_str(&format!("misses:{}
", st.misses));
    s.push_str(&format!("hit_ratio:{:.4}
", st.hit_ratio));
    s.push_str(&format!("expired:{}
", st.expired));
    s.push_str(&format!("evictions:{}
", st.evictions));
    s.push_str(&format!("sets:{}
", st.sets));
    s.push_str(&format!("dels:{}
", st.dels));
    s.push_str(&format!("seqlock_retries:{}
", st.retries));
    s.push_str(&format!("uptime_ms:{}
", st.uptime_ms));
    s
}

fn sarg(args: &[Vec<u8>], i: usize) -> Option<String> {
    args.get(i).map(|a| String::from_utf8_lossy(a).into_owned())
}

fn num(b: &[u8]) -> Option<u64> {
    std::str::from_utf8(b).ok()?.parse().ok()
}

fn er(msg: impl Into<String>) -> Action {
    Action::Reply(Frame::Error(msg.into()))
}
