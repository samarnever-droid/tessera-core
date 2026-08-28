//! Persistent Consumer Groups & Pending Entries List (PEL).

use crate::stream::ring::{Stream, StreamEntry, StreamId};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingEntry {
    pub id: StreamId,
    pub consumer: String,
    pub delivery_time_ms: u64,
    pub delivery_count: u32,
}

pub struct ConsumerGroup {
    pub name: String,
    pub last_delivered_id: StreamId,
    pub pel: BTreeMap<StreamId, PendingEntry>,
    pub consumers: HashMap<String, u64>, // consumer_name -> last_active_time_ms
}

impl ConsumerGroup {
    pub fn new(name: impl Into<String>, start_id: StreamId) -> Self {
        Self {
            name: name.into(),
            last_delivered_id: start_id,
            pel: BTreeMap::new(),
            consumers: HashMap::new(),
        }
    }

    /// Reads unread messages for a given consumer and places them in the PEL.
    pub fn read_group(
        &mut self,
        consumer: &str,
        count: usize,
        stream: &Stream,
    ) -> Vec<StreamEntry> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.consumers.insert(consumer.to_string(), now_ms);

        let unread = stream.range(Some(StreamId::new(self.last_delivered_id.timestamp_ms, self.last_delivered_id.sequence + 1)), None);
        let mut delivered = Vec::new();

        for entry in unread.into_iter().take(count) {
            self.last_delivered_id = entry.id;
            self.pel.insert(
                entry.id,
                PendingEntry {
                    id: entry.id,
                    consumer: consumer.to_string(),
                    delivery_time_ms: now_ms,
                    delivery_count: 1,
                },
            );
            delivered.push(entry);
        }

        delivered
    }

    /// Acknowledges processed messages and clears them from the PEL.
    pub fn ack(&mut self, ids: &[StreamId]) -> usize {
        let mut acked = 0;
        for id in ids {
            if self.pel.remove(id).is_some() {
                acked += 1;
            }
        }
        acked
    }

    /// Claims timed-out pending entries and transfers ownership to a new consumer.
    pub fn claim(
        &mut self,
        ids: &[StreamId],
        new_consumer: &str,
        min_idle_time_ms: u64,
    ) -> Vec<StreamId> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut claimed = Vec::new();

        for id in ids {
            if let Some(entry) = self.pel.get_mut(id) {
                if now_ms >= entry.delivery_time_ms + min_idle_time_ms {
                    entry.consumer = new_consumer.to_string();
                    entry.delivery_time_ms = now_ms;
                    entry.delivery_count += 1;
                    claimed.push(*id);
                }
            }
        }

        claimed
    }

    pub fn pending_count(&self) -> usize {
        self.pel.len()
    }
}
