# Python分布式系统设计：从理论到实践

## 哲学思考：分布式系统的本质

分布式系统的设计哲学源于一个根本性问题：如何在不可靠的网络环境下构建可靠的系统？这个问题触及了计算机科学的核心——**一致性、可用性、分区容忍性**的权衡。

> "分布式系统就是这样一个系统：你不知道一台机器什么时候会宕机，但你必须保证系统继续工作。" —— Leslie Lamport

## 理论基础：CAP定理与一致性模型

### CAP定理的深刻理解

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
import asyncio
import time
import random
from dataclasses import dataclass
from typing import Set, Tuple
import hashlib
import json
import logging

class ConsistencyLevel(Enum):
    """一致性级别枚举"""
    STRONG = "strong"           # 强一致性
    EVENTUAL = "eventual"       # 最终一致性
    CAUSAL = "causal"          # 因果一致性
    SESSION = "session"        # 会话一致性

class CAPChoice(Enum):
    """CAP定理选择枚举"""
    CP = "CP"  # 一致性 + 分区容忍性
    AP = "AP"  # 可用性 + 分区容忍性
    CA = "CA"  # 一致性 + 可用性（理论上的，实际很难实现）

@dataclass
class SystemState:
    """系统状态"""
    available: bool = True
    consistent: bool = True
    partition_tolerant: bool = True
    network_partition: bool = False
    failed_nodes: Set[str] = None

    def __post_init__(self):
        if self.failed_nodes is None:
            self.failed_nodes = set()

class DistributedSystemPhilosophy:
    """分布式系统设计哲学"""

    def __init__(self, cap_choice: CAPChoice):
        self.cap_choice = cap_choice
        self.logger = logging.getLogger(__name__)

    def analyze_tradeoffs(self) -> Dict[str, Any]:
        """分析CAP权衡"""
        tradeoffs = {
            CAPChoice.CP: {
                "strengths": ["强一致性", "数据完整性"],
                "weaknesses": ["可用性降低", "延迟增加"],
                "use_cases": ["银行系统", "数据库事务"]
            },
            CAPChoice.AP: {
                "strengths": ["高可用性", "低延迟"],
                "weaknesses": ["可能的数据不一致", "需要冲突解决"],
                "use_cases": ["社交媒体", "内容分发网络"]
            },
            CAPChoice.CA: {
                "strengths": ["理想状态", "完美体验"],
                "weaknesses": ["理论上可行", "实际难以实现"],
                "use_cases": ["单机系统", "本地网络"]
            }
        }
        return tradeoffs[self.cap_choice]

    def make_design_decisions(self, requirements: Dict[str, Any]) -> List[str]:
        """基于需求做出设计决策"""
        decisions = []

        if requirements.get("consistency_critical", False):
            decisions.append("采用强一致性算法如Raft或Paxos")
            decisions.append("实现事务性操作")

        if requirements.get("high_availability", False):
            decisions.append("实现冗余备份")
            decisions.append("采用故障转移机制")

        if requirements.get("low_latency", False):
            decisions.append("采用最终一致性")
            decisions.append("实现本地缓存")

        return decisions
```

## 一致性协议的实现

### Raft共识算法详解

```python
class RaftNodeState(Enum):
    """Raft节点状态"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

@dataclass
class LogEntry:
    """日志条目"""
    term: int
    command: Any
    index: int

class RaftNode:
    """Raft共识算法节点实现"""

    def __init__(self, node_id: str, all_nodes: List[str]):
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.state = RaftNodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        self.leader_id: Optional[str] = None

        # 领导者特有状态
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}

        # 定时器
        self.election_timeout = random.uniform(0.15, 0.3)
        self.heartbeat_timeout = 0.05
        self.last_heartbeat = time.time()
        self.last_election = time.time()

        # 网络和存储
        self.network = NetworkSimulator()
        self.storage = PersistentStorage()

        self.logger = logging.getLogger(f"RaftNode-{node_id}")
        self._load_state()

    def _load_state(self):
        """从持久化存储加载状态"""
        state = self.storage.load_state()
        if state:
            self.current_term = state.get('current_term', 0)
            self.voted_for = state.get('voted_for')
            self.log = [LogEntry(**entry) for entry in state.get('log', [])]

    def _save_state(self):
        """保存状态到持久化存储"""
        state = {
            'current_term': self.current_term,
            'voted_for': self.voted_for,
            'log': [{'term': entry.term, 'command': entry.command, 'index': entry.index}
                   for entry in self.log]
        }
        self.storage.save_state(state)

    async def start(self):
        """启动Raft节点"""
        self.logger.info(f"Starting Raft node {self.node_id}")
        while True:
            await self._tick()
            await asyncio.sleep(0.01)  # 10ms tick

    async def _tick(self):
        """主循环tick"""
        current_time = time.time()

        if self.state == RaftNodeState.LEADER:
            await self._leader_tick(current_time)
        else:
            await self._follower_tick(current_time)

    async def _follower_tick(self, current_time: float):
        """Follower状态处理"""
        # 检查选举超时
        if current_time - self.last_heartbeat > self.election_timeout:
            await self._start_election()

    async def _leader_tick(self, current_time: float):
        """Leader状态处理"""
        # 发送心跳
        if current_time - self.last_heartbeat > self.heartbeat_timeout:
            await self._send_heartbeats()
            self.last_heartbeat = current_time

    async def _start_election(self):
        """开始选举"""
        self.logger.info(f"Starting election for term {self.current_term + 1}")

        self.state = RaftNodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_election = time.time()

        self._save_state()

        # 发送投票请求
        votes = 1  # 自己投自己一票
        requests = []

        for node_id in self.all_nodes:
            if node_id != self.node_id:
                request = self._request_vote(node_id)
                requests.append(request)

        # 等待投票结果
        results = await asyncio.gather(*requests, return_exceptions=True)

        for result in results:
            if isinstance(result, bool) and result:
                votes += 1

        # 检查是否获得多数票
        if votes > len(self.all_nodes) // 2:
            await self._become_leader()
        else:
            self.state = RaftNodeState.FOLLOWER
            self.last_heartbeat = time.time()

    async def _request_vote(self, node_id: str) -> bool:
        """请求投票"""
        try:
            last_log_index = len(self.log) - 1 if self.log else -1
            last_log_term = self.log[-1].term if self.log else 0

            request = {
                'term': self.current_term,
                'candidate_id': self.node_id,
                'last_log_index': last_log_index,
                'last_log_term': last_log_term
            }

            response = await self.network.send_request(node_id, 'request_vote', request)
            return response.get('vote_granted', False)
        except Exception as e:
            self.logger.error(f"Failed to request vote from {node_id}: {e}")
            return False

    async def _become_leader(self):
        """成为领导者"""
        self.logger.info(f"Becoming leader for term {self.current_term}")
        self.state = RaftNodeState.LEADER
        self.leader_id = self.node_id

        # 初始化领导者状态
        for node_id in self.all_nodes:
            if node_id != self.node_id:
                self.next_index[node_id] = len(self.log)
                self.match_index[node_id] = 0

        # 发送初始心跳
        await self._send_heartbeats()

    async def _send_heartbeats(self):
        """发送心跳"""
        for node_id in self.all_nodes:
            if node_id != self.node_id:
                asyncio.create_task(self._send_append_entries(node_id))

    async def _send_append_entries(self, node_id: str):
        """发送日志条目"""
        try:
            prev_log_index = self.next_index[node_id] - 1
            prev_log_term = self.log[prev_log_index].term if prev_log_index >= 0 else 0

            entries = self.log[self.next_index[node_id]:]

            request = {
                'term': self.current_term,
                'leader_id': self.node_id,
                'prev_log_index': prev_log_index,
                'prev_log_term': prev_log_term,
                'entries': [{'term': entry.term, 'command': entry.command, 'index': entry.index}
                           for entry in entries],
                'leader_commit': self.commit_index
            }

            response = await self.network.send_request(node_id, 'append_entries', request)

            if response.get('success', False):
                self.next_index[node_id] += len(entries)
                self.match_index[node_id] = self.next_index[node_id] - 1
                await self._update_commit_index()
            else:
                self.next_index[node_id] = max(0, self.next_index[node_id] - 1)

        except Exception as e:
            self.logger.error(f"Failed to send append entries to {node_id}: {e}")

    async def _update_commit_index(self):
        """更新提交索引"""
        # 找到被大多数节点复制的日志条目
        for i in range(len(self.log) - 1, -1, -1):
            if self.log[i].term == self.current_term:
                count = 1  # leader自己
                for node_id in self.all_nodes:
                    if node_id != self.node_id and self.match_index.get(node_id, 0) >= i:
                        count += 1

                if count > len(self.all_nodes) // 2:
                    self.commit_index = i
                    break

    async def propose_command(self, command: Any) -> bool:
        """提议新命令"""
        if self.state != RaftNodeState.LEADER:
            return False

        # 创建新的日志条目
        entry = LogEntry(
            term=self.current_term,
            command=command,
            index=len(self.log)
        )

        self.log.append(entry)
        self._save_state()

        # 复制到其他节点
        await self._send_heartbeats()

        return True
```

## 分布式数据管理

### 分片与复制策略

```python
class ShardingStrategy:
    """分片策略抽象类"""

    @abstractmethod
    def get_shard(self, key: Any, total_shards: int) -> int:
        """根据键获取分片ID"""
        pass

class HashSharding(ShardingStrategy):
    """哈希分片策略"""

    def __init__(self, hash_func=None):
        self.hash_func = hash_func or self._default_hash

    def _default_hash(self, key: Any) -> int:
        """默认哈希函数"""
        if isinstance(key, str):
            return int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash(key)

    def get_shard(self, key: Any, total_shards: int) -> int:
        return self.hash_func(key) % total_shards

class RangeSharding(ShardingStrategy):
    """范围分片策略"""

    def __init__(self, ranges: List[Tuple[Any, Any]]):
        self.ranges = ranges

    def get_shard(self, key: Any, total_shards: int) -> int:
        for i, (start, end) in enumerate(self.ranges):
            if start <= key < end:
                return i
        return total_shards - 1  # 默认最后一个分片

class ConsistentHashSharding(ShardingStrategy):
    """一致性哈希分片"""

    def __init__(self, virtual_nodes: int = 3):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.nodes: Set[str] = set()

    def add_node(self, node: str):
        """添加节点"""
        self.nodes.add(node)
        for i in range(self.virtual_nodes):
            virtual_node_key = f"{node}-{i}"
            hash_key = self._hash_key(virtual_node_key)
            self.ring[hash_key] = node

    def remove_node(self, node: str):
        """移除节点"""
        self.nodes.remove(node)
        for i in range(self.virtual_nodes):
            virtual_node_key = f"{node}-{i}"
            hash_key = self._hash_key(virtual_node_key)
            del self.ring[hash_key]

    def _hash_key(self, key: str) -> int:
        """哈希键"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def get_shard(self, key: Any, total_shards: int) -> int:
        if not self.ring:
            return 0

        hash_key = self._hash_key(str(key))

        # 找到第一个大于等于hash_key的节点
        ring_keys = sorted(self.ring.keys())
        for ring_key in ring_keys:
            if hash_key <= ring_key:
                return list(self.nodes).index(self.ring[ring_key])

        # 如果没有找到，返回第一个节点
        return list(self.nodes).index(self.ring[ring_keys[0]])

class ReplicationManager:
    """复制管理器"""

    def __init__(self, replication_factor: int):
        self.replication_factor = replication_factor
        self.logger = logging.getLogger(__name__)

    async def write(self, key: Any, value: Any, nodes: List[str],
                   quorum: Optional[int] = None) -> bool:
        """写入数据"""
        if quorum is None:
            quorum = (self.replication_factor // 2) + 1

        success_count = 0
        results = []

        for node in nodes:
            result = asyncio.create_task(self._write_to_node(node, key, value))
            results.append(result)

        # 等待足够数量的成功响应
        completed, pending = await asyncio.wait(
            results,
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in completed:
            if task.result():
                success_count += 1
                if success_count >= quorum:
                    # 取消剩余的任务
                    for p in pending:
                        p.cancel()
                    return True

        return False

    async def read(self, key: Any, nodes: List[str],
                  quorum: Optional[int] = None) -> Any:
        """读取数据"""
        if quorum is None:
            quorum = (self.replication_factor // 2) + 1

        results = []

        for node in nodes:
            result = asyncio.create_task(self._read_from_node(node, key))
            results.append(result)

        # 等待足够数量的响应
        completed, pending = await asyncio.wait(
            results,
            return_when=asyncio.FIRST_COMPLETED
        )

        # 收集结果并选择最新版本
        values = []
        for task in completed:
            result = task.result()
            if result is not None:
                values.append(result)

        # 取消剩余的任务
        for p in pending:
            p.cancel()

        if not values:
            return None

        # 简单的版本冲突解决：选择最新时间戳
        return max(values, key=lambda x: x.get('timestamp', 0))

    async def _write_to_node(self, node: str, key: Any, value: Any) -> bool:
        """向节点写入数据"""
        try:
            # 模拟网络延迟
            await asyncio.sleep(random.uniform(0.01, 0.1))

            # 这里应该是实际的网络请求
            # 简化实现，总是返回成功
            return True
        except Exception as e:
            self.logger.error(f"Failed to write to node {node}: {e}")
            return False

    async def _read_from_node(self, node: str, key: Any) -> Any:
        """从节点读取数据"""
        try:
            # 模拟网络延迟
            await asyncio.sleep(random.uniform(0.01, 0.1))

            # 这里应该是实际的网络请求
            # 简化实现，返回模拟数据
            return {
                'value': f'value_for_{key}',
                'timestamp': time.time(),
                'node': node
            }
        except Exception as e:
            self.logger.error(f"Failed to read from node {node}: {e}")
            return None

class DistributedDataStore:
    """分布式数据存储"""

    def __init__(self, nodes: List[str], sharding_strategy: ShardingStrategy,
                 replication_factor: int = 3):
        self.nodes = nodes
        self.sharding_strategy = sharding_strategy
        self.replication_manager = ReplicationManager(replication_factor)
        self.logger = logging.getLogger(__name__)

    async def put(self, key: Any, value: Any) -> bool:
        """存储键值对"""
        # 确定主分片
        primary_shard = self.sharding_strategy.get_shard(key, len(self.nodes))

        # 选择复制节点（简化版：选择相邻节点）
        replica_nodes = self._select_replica_nodes(primary_shard)

        # 写入数据
        success = await self.replication_manager.write(key, value, replica_nodes)

        if success:
            self.logger.info(f"Successfully stored key {key} in shard {primary_shard}")
        else:
            self.logger.error(f"Failed to store key {key}")

        return success

    async def get(self, key: Any) -> Any:
        """获取值"""
        # 确定主分片
        primary_shard = self.sharding_strategy.get_shard(key, len(self.nodes))

        # 选择复制节点
        replica_nodes = self._select_replica_nodes(primary_shard)

        # 读取数据
        result = await self.replication_manager.read(key, replica_nodes)

        if result:
            return result.get('value')
        return None

    def _select_replica_nodes(self, primary_shard: int) -> List[str]:
        """选择复制节点"""
        nodes = []
        replication_factor = self.replication_manager.replication_factor

        for i in range(replication_factor):
            shard = (primary_shard + i) % len(self.nodes)
            nodes.append(self.nodes[shard])

        return nodes
```

## 分布式事务与一致性

### 两阶段提交协议

```python
class TransactionStatus(Enum):
    """事务状态"""
    ACTIVE = "active"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"

class TwoPhaseCommitCoordinator:
    """两阶段提交协调器"""

    def __init__(self, participants: List[str]):
        self.participants = participants
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

    async def begin_transaction(self, transaction_id: str) -> bool:
        """开始事务"""
        if transaction_id in self.active_transactions:
            return False

        self.active_transactions[transaction_id] = {
            'status': TransactionStatus.ACTIVE,
            'participants': set(),
            'votes': {},
            'start_time': time.time()
        }

        self.logger.info(f"Started transaction {transaction_id}")
        return True

    async def prepare(self, transaction_id: str) -> bool:
        """准备阶段"""
        if transaction_id not in self.active_transactions:
            return False

        transaction = self.active_transactions[transaction_id]
        transaction['status'] = TransactionStatus.PREPARED

        # 向所有参与者发送准备请求
        votes = {}
        prepare_tasks = []

        for participant in self.participants:
            task = asyncio.create_task(
                self._send_prepare_request(participant, transaction_id)
            )
            prepare_tasks.append((participant, task))

        # 收集投票结果
        for participant, task in prepare_tasks:
            try:
                vote = await asyncio.wait_for(task, timeout=5.0)
                votes[participant] = vote
                if vote:
                    transaction['participants'].add(participant)
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout waiting for vote from {participant}")
                votes[participant] = False
            except Exception as e:
                self.logger.error(f"Error getting vote from {participant}: {e}")
                votes[participant] = False

        transaction['votes'] = votes

        # 检查是否所有参与者都同意
        if all(votes.values()):
            await self._commit(transaction_id)
            return True
        else:
            await self._abort(transaction_id)
            return False

    async def _send_prepare_request(self, participant: str, transaction_id: str) -> bool:
        """发送准备请求"""
        try:
            # 模拟网络请求
            await asyncio.sleep(random.uniform(0.1, 0.3))

            # 模拟参与者投票（90%成功率）
            return random.random() < 0.9
        except Exception as e:
            self.logger.error(f"Failed to send prepare to {participant}: {e}")
            return False

    async def _commit(self, transaction_id: str):
        """提交事务"""
        if transaction_id not in self.active_transactions:
            return

        transaction = self.active_transactions[transaction_id]
        transaction['status'] = TransactionStatus.COMMITTED

        # 向所有参与者发送提交请求
        commit_tasks = []

        for participant in transaction['participants']:
            task = asyncio.create_task(
                self._send_commit_request(participant, transaction_id)
            )
            commit_tasks.append(task)

        # 等待所有确认
        await asyncio.gather(*commit_tasks, return_exceptions=True)

        self.logger.info(f"Committed transaction {transaction_id}")

        # 清理事务
        del self.active_transactions[transaction_id]

    async def _abort(self, transaction_id: str):
        """中止事务"""
        if transaction_id not in self.active_transactions:
            return

        transaction = self.active_transactions[transaction_id]
        transaction['status'] = TransactionStatus.ABORTED

        # 向所有参与者发送中止请求
        abort_tasks = []

        for participant in transaction['participants']:
            task = asyncio.create_task(
                self._send_abort_request(participant, transaction_id)
            )
            abort_tasks.append(task)

        # 等待所有确认
        await asyncio.gather(*abort_tasks, return_exceptions=True)

        self.logger.info(f"Aborted transaction {transaction_id}")

        # 清理事务
        del self.active_transactions[transaction_id]

    async def _send_commit_request(self, participant: str, transaction_id: str):
        """发送提交请求"""
        try:
            await asyncio.sleep(random.uniform(0.1, 0.2))
            self.logger.debug(f"Sent commit to {participant} for {transaction_id}")
        except Exception as e:
            self.logger.error(f"Failed to send commit to {participant}: {e}")

    async def _send_abort_request(self, participant: str, transaction_id: str):
        """发送中止请求"""
        try:
            await asyncio.sleep(random.uniform(0.1, 0.2))
            self.logger.debug(f"Sent abort to {participant} for {transaction_id}")
        except Exception as e:
            self.logger.error(f"Failed to send abort to {participant}: {e}")

class SagaPattern:
    """Saga模式实现"""

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.compensations: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def add_step(self, name: str, action: callable, compensation: callable):
        """添加步骤"""
        self.steps.append({
            'name': name,
            'action': action,
            'compensation': compensation
        })

    async def execute(self, context: Dict[str, Any]) -> bool:
        """执行Saga"""
        executed_steps = []

        try:
            for step in self.steps:
                self.logger.info(f"Executing step: {step['name']}")

                # 执行步骤
                result = await step['action'](context)
                if not result:
                    raise Exception(f"Step {step['name']} failed")

                executed_steps.append(step)

        except Exception as e:
            self.logger.error(f"Saga failed: {e}")

            # 执行补偿
            for step in reversed(executed_steps):
                try:
                    self.logger.info(f"Compensating step: {step['name']}")
                    await step['compensation'](context)
                except Exception as comp_error:
                    self.logger.error(f"Compensation failed for {step['name']}: {comp_error}")

            return False

        return True
```

## 分布式锁与同步

### 分布式锁实现

```python
class DistributedLock:
    """分布式锁实现"""

    def __init__(self, lock_name: str, nodes: List[str],
                 ttl: float = 30.0, retry_interval: float = 0.1):
        self.lock_name = lock_name
        self.nodes = nodes
        self.ttl = ttl
        self.retry_interval = retry_interval
        self.owner = None
        self.acquired_at = None
        self.logger = logging.getLogger(__name__)

    async def acquire(self, owner: str, timeout: Optional[float] = None) -> bool:
        """获取锁"""
        start_time = time.time()

        while True:
            # 尝试在所有节点上获取锁
            success_count = 0
            acquire_tasks = []

            for node in self.nodes:
                task = asyncio.create_task(
                    self._try_acquire_lock(node, owner)
                )
                acquire_tasks.append(task)

            # 等待所有节点响应
            results = await asyncio.gather(*acquire_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, bool) and result:
                    success_count += 1

            # 检查是否获得多数节点支持
            if success_count > len(self.nodes) // 2:
                self.owner = owner
                self.acquired_at = time.time()

                # 启动续约任务
                asyncio.create_task(self._renew_lease())

                self.logger.info(f"Lock {self.lock_name} acquired by {owner}")
                return True

            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(f"Failed to acquire lock {self.lock_name} within timeout")
                return False

            # 等待重试
            await asyncio.sleep(self.retry_interval)

    async def release(self, owner: str) -> bool:
        """释放锁"""
        if self.owner != owner:
            return False

        # 向所有节点发送释放请求
        release_tasks = []

        for node in self.nodes:
            task = asyncio.create_task(
                self._release_lock(node, owner)
            )
            release_tasks.append(task)

        # 等待所有节点响应
        results = await asyncio.gather(*release_tasks, return_exceptions=True)

        success_count = sum(1 for result in results if isinstance(result, bool) and result)

        if success_count > len(self.nodes) // 2:
            self.owner = None
            self.acquired_at = None
            self.logger.info(f"Lock {self.lock_name} released by {owner}")
            return True

        return False

    async def _try_acquire_lock(self, node: str, owner: str) -> bool:
        """尝试在单个节点上获取锁"""
        try:
            # 模拟网络请求
            await asyncio.sleep(random.uniform(0.01, 0.05))

            # 这里应该是实际的网络请求
            # 简化实现，模拟锁获取
            return True
        except Exception as e:
            self.logger.error(f"Failed to acquire lock on {node}: {e}")
            return False

    async def _release_lock(self, node: str, owner: str) -> bool:
        """在单个节点上释放锁"""
        try:
            await asyncio.sleep(random.uniform(0.01, 0.05))
            return True
        except Exception as e:
            self.logger.error(f"Failed to release lock on {node}: {e}")
            return False

    async def _renew_lease(self):
        """续约锁租约"""
        while self.owner is not None:
            await asyncio.sleep(self.ttl * 0.7)  # 在TTL的70%时续约

            if self.owner is None:
                break

            # 续约锁
            success_count = 0
            renew_tasks = []

            for node in self.nodes:
                task = asyncio.create_task(
                    self._renew_lock(node, self.owner)
                )
                renew_tasks.append(task)

            results = await asyncio.gather(*renew_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, bool) and result:
                    success_count += 1

            # 如果续约失败，释放锁
            if success_count <= len(self.nodes) // 2:
                self.logger.warning("Failed to renew lease, releasing lock")
                await self.release(self.owner)
                break

class AtomicCounter:
    """原子计数器"""

    def __init__(self, name: str, nodes: List[str]):
        self.name = name
        self.nodes = nodes
        self.logger = logging.getLogger(__name__)

    async def increment(self, delta: int = 1) -> int:
        """原子递增"""
        # 使用CAS（Compare-And-Swap）模式
        while True:
            # 获取当前值
            current_value = await self._get_value()

            # 计算新值
            new_value = current_value + delta

            # 尝试设置新值
            success = await self._compare_and_swap(current_value, new_value)

            if success:
                return new_value

            # 如果失败，重试
            await asyncio.sleep(0.01)

    async def _get_value(self) -> int:
        """获取当前值"""
        # 从多数节点获取值
        values = []
        get_tasks = []

        for node in self.nodes:
            task = asyncio.create_task(self._get_from_node(node))
            get_tasks.append(task)

        results = await asyncio.gather(*get_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, int):
                values.append(result)

        # 返回最大值（最新的）
        return max(values) if values else 0

    async def _compare_and_swap(self, old_value: int, new_value: int) -> bool:
        """比较并交换"""
        success_count = 0
        cas_tasks = []

        for node in self.nodes:
            task = asyncio.create_task(
                self._cas_on_node(node, old_value, new_value)
            )
            cas_tasks.append(task)

        results = await asyncio.gather(*cas_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, bool) and result:
                success_count += 1

        return success_count > len(self.nodes) // 2

    async def _get_from_node(self, node: str) -> int:
        """从节点获取值"""
        try:
            await asyncio.sleep(random.uniform(0.01, 0.05))
            # 模拟返回值
            return 0
        except Exception as e:
            self.logger.error(f"Failed to get value from {node}: {e}")
            raise

    async def _cas_on_node(self, node: str, old_value: int, new_value: int) -> bool:
        """在节点上执行CAS"""
        try:
            await asyncio.sleep(random.uniform(0.01, 0.05))
            # 模拟CAS操作
            return True
        except Exception as e:
            self.logger.error(f"Failed to CAS on {node}: {e}")
            return False
```

## 故障检测与恢复

### 心跳与健康检查

```python
class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class FailureDetector:
    """故障检测器"""

    def __init__(self, nodes: List[str], heartbeat_interval: float = 5.0,
                 timeout_threshold: float = 15.0):
        self.nodes = nodes
        self.heartbeat_interval = heartbeat_interval
        self.timeout_threshold = timeout_threshold
        self.node_status: Dict[str, HealthStatus] = {}
        self.last_heartbeat: Dict[str, float] = {}
        self.failure_count: Dict[str, int] = {}

        # 初始化状态
        for node in nodes:
            self.node_status[node] = HealthStatus.UNKNOWN
            self.last_heartbeat[node] = time.time()
            self.failure_count[node] = 0

        self.logger = logging.getLogger(__name__)

    async def start(self):
        """启动故障检测"""
        self.logger.info("Starting failure detector")

        while True:
            await self._check_nodes()
            await asyncio.sleep(self.heartbeat_interval)

    async def _check_nodes(self):
        """检查所有节点"""
        check_tasks = []

        for node in self.nodes:
            task = asyncio.create_task(self._check_node_health(node))
            check_tasks.append(task)

        await asyncio.gather(*check_tasks, return_exceptions=True)

    async def _check_node_health(self, node: str):
        """检查单个节点健康状态"""
        try:
            # 发送心跳请求
            healthy = await self._send_heartbeat(node)

            if healthy:
                self.last_heartbeat[node] = time.time()
                self.failure_count[node] = 0
                self.node_status[node] = HealthStatus.HEALTHY
            else:
                self.failure_count[node] += 1

                if self.failure_count[node] >= 3:
                    self.node_status[node] = HealthStatus.UNHEALTHY
                    self.logger.warning(f"Node {node} marked as unhealthy")

        except Exception as e:
            self.failure_count[node] += 1
            self.logger.error(f"Failed to check node {node}: {e}")

            if self.failure_count[node] >= 3:
                self.node_status[node] = HealthStatus.UNHEALTHY

    async def _send_heartbeat(self, node: str) -> bool:
        """发送心跳"""
        try:
            await asyncio.sleep(random.uniform(0.01, 0.1))
            # 模拟心跳成功（95%成功率）
            return random.random() < 0.95
        except Exception as e:
            self.logger.error(f"Heartbeat failed for {node}: {e}")
            return False

    def get_healthy_nodes(self) -> List[str]:
        """获取健康节点列表"""
        return [
            node for node, status in self.node_status.items()
            if status == HealthStatus.HEALTHY
        ]

    def is_node_healthy(self, node: str) -> bool:
        """检查节点是否健康"""
        return self.node_status.get(node, HealthStatus.UNKNOWN) == HealthStatus.HEALTHY

class CircuitBreaker:
    """断路器模式"""

    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger(__name__)

    async def call(self, func: callable, *args, **kwargs) -> Any:
        """调用函数，带有断路器保护"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)

            # 成功，重置状态
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                self.logger.info("Circuit breaker moved to CLOSED state")

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

            raise e

class RetryMechanism:
    """重试机制"""

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0,
                 max_delay: float = 30.0, backoff_factor: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    async def retry(self, func: callable, *args, **kwargs) -> Any:
        """带重试的函数调用"""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.max_attempts - 1:
                    delay = min(
                        self.base_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    await asyncio.sleep(delay)

        raise last_exception
```

## 系统监控与指标

```python
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class Metric:
    """指标数据"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """指标收集器"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.logger = logging.getLogger(__name__)

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """记录指标"""
        metric = Metric(name, value, tags=tags or {})
        self.metrics[name].append(metric)

    def get_metrics(self, name: str, time_range: Optional[float] = None) -> List[Metric]:
        """获取指标"""
        if name not in self.metrics:
            return []

        metrics = list(self.metrics[name])

        if time_range:
            cutoff_time = time.time() - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        return metrics

    def get_average(self, name: str, time_range: Optional[float] = None) -> float:
        """获取平均值"""
        metrics = self.get_metrics(name, time_range)
        if not metrics:
            return 0.0

        return statistics.mean(m.value for m in metrics)

    def get_percentile(self, name: str, percentile: float,
                      time_range: Optional[float] = None) -> float:
        """获取百分位数"""
        metrics = self.get_metrics(name, time_range)
        if not metrics:
            return 0.0

        values = sorted(m.value for m in metrics)
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]

    def get_rate(self, name: str, time_range: float = 60.0) -> float:
        """获取速率"""
        metrics = self.get_metrics(name, time_range)
        if len(metrics) < 2:
            return 0.0

        # 计算时间范围内的变化率
        start_value = metrics[0].value
        end_value = metrics[-1].value
        time_diff = metrics[-1].timestamp - metrics[0].timestamp

        if time_diff == 0:
            return 0.0

        return (end_value - start_value) / time_diff

class SystemMonitor:
    """系统监控器"""

    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.metrics_collector = MetricsCollector()
        self.failure_detector = FailureDetector(nodes)
        self.alert_manager = AlertManager()
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """启动监控"""
        self.logger.info("Starting system monitor")

        # 启动故障检测
        asyncio.create_task(self.failure_detector.start())

        # 启动指标收集
        asyncio.create_task(self._collect_metrics())

        # 启动健康检查
        asyncio.create_task(self._health_checks())

    async def _collect_metrics(self):
        """收集指标"""
        while True:
            try:
                # 收集系统指标
                await self._collect_system_metrics()

                # 收集应用指标
                await self._collect_application_metrics()

                # 收集网络指标
                await self._collect_network_metrics()

                # 检查告警
                await self._check_alerts()

                await asyncio.sleep(10)  # 每10秒收集一次

            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(5)

    async def _collect_system_metrics(self):
        """收集系统指标"""
        for node in self.nodes:
            if not self.failure_detector.is_node_healthy(node):
                continue

            try:
                # 模拟收集系统指标
                cpu_usage = random.uniform(10, 80)
                memory_usage = random.uniform(30, 90)
                disk_usage = random.uniform(20, 85)

                # 记录指标
                self.metrics_collector.record_metric(
                    f"system.cpu_usage",
                    cpu_usage,
                    {"node": node}
                )
                self.metrics_collector.record_metric(
                    f"system.memory_usage",
                    memory_usage,
                    {"node": node}
                )
                self.metrics_collector.record_metric(
                    f"system.disk_usage",
                    disk_usage,
                    {"node": node}
                )

            except Exception as e:
                self.logger.error(f"Failed to collect system metrics from {node}: {e}")

    async def _collect_application_metrics(self):
        """收集应用指标"""
        for node in self.nodes:
            if not self.failure_detector.is_node_healthy(node):
                continue

            try:
                # 模拟收集应用指标
                request_rate = random.uniform(100, 1000)
                response_time = random.uniform(0.01, 0.5)
                error_rate = random.uniform(0, 0.05)

                self.metrics_collector.record_metric(
                    f"app.request_rate",
                    request_rate,
                    {"node": node}
                )
                self.metrics_collector.record_metric(
                    f"app.response_time",
                    response_time,
                    {"node": node}
                )
                self.metrics_collector.record_metric(
                    f"app.error_rate",
                    error_rate,
                    {"node": node}
                )

            except Exception as e:
                self.logger.error(f"Failed to collect app metrics from {node}: {e}")

    async def _collect_network_metrics(self):
        """收集网络指标"""
        for node in self.nodes:
            if not self.failure_detector.is_node_healthy(node):
                continue

            try:
                # 模拟收集网络指标
                network_latency = random.uniform(0.001, 0.1)
                throughput = random.uniform(1000, 10000)

                self.metrics_collector.record_metric(
                    f"network.latency",
                    network_latency,
                    {"node": node}
                )
                self.metrics_collector.record_metric(
                    f"network.throughput",
                    throughput,
                    {"node": node}
                )

            except Exception as e:
                self.logger.error(f"Failed to collect network metrics from {node}: {e}")

    async def _health_checks(self):
        """健康检查"""
        while True:
            try:
                # 检查系统健康状态
                await self._check_system_health()

                # 检查应用健康状态
                await self._check_application_health()

                await asyncio.sleep(30)  # 每30秒检查一次

            except Exception as e:
                self.logger.error(f"Error in health checks: {e}")
                await asyncio.sleep(10)

    async def _check_system_health(self):
        """检查系统健康状态"""
        for node in self.nodes:
            cpu_avg = self.metrics_collector.get_average(
                f"system.cpu_usage", 300, {"node": node}
            )
            memory_avg = self.metrics_collector.get_average(
                f"system.memory_usage", 300, {"node": node}
            )

            # 检查CPU使用率
            if cpu_avg > 80:
                await self.alert_manager.send_alert(
                    "HIGH_CPU_USAGE",
                    f"High CPU usage on {node}: {cpu_avg:.1f}%",
                    {"node": node, "cpu_usage": cpu_avg}
                )

            # 检查内存使用率
            if memory_avg > 90:
                await self.alert_manager.send_alert(
                    "HIGH_MEMORY_USAGE",
                    f"High memory usage on {node}: {memory_avg:.1f}%",
                    {"node": node, "memory_usage": memory_avg}
                )

    async def _check_application_health(self):
        """检查应用健康状态"""
        for node in self.nodes:
            error_rate = self.metrics_collector.get_average(
                f"app.error_rate", 300, {"node": node}
            )
            response_time = self.metrics_collector.get_percentile(
                f"app.response_time", 95, 300, {"node": node}
            )

            # 检查错误率
            if error_rate > 0.01:  # 1%
                await self.alert_manager.send_alert(
                    "HIGH_ERROR_RATE",
                    f"High error rate on {node}: {error_rate:.2%}",
                    {"node": node, "error_rate": error_rate}
                )

            # 检查响应时间
            if response_time > 0.5:  # 500ms
                await self.alert_manager.send_alert(
                    "HIGH_RESPONSE_TIME",
                    f"High response time on {node}: {response_time:.3f}s",
                    {"node": node, "response_time": response_time}
                )

    async def _check_alerts(self):
        """检查告警条件"""
        # 这里可以添加更多复杂的告警逻辑
        pass

class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_suppressions: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)

    async def send_alert(self, alert_type: str, message: str,
                        context: Dict[str, Any] = None):
        """发送告警"""
        # 检查告警抑制
        suppression_key = f"{alert_type}_{context.get('node', '')}"

        if suppression_key in self.alert_suppressions:
            if time.time() - self.alert_suppressions[suppression_key] < 300:  # 5分钟抑制
                return

        # 记录告警
        alert = {
            'type': alert_type,
            'message': message,
            'context': context or {},
            'timestamp': time.time(),
            'severity': self._get_severity(alert_type)
        }

        self.alert_history.append(alert)

        # 发送告警通知
        await self._send_notification(alert)

        # 设置告警抑制
        self.alert_suppressions[suppression_key] = time.time()

        self.logger.warning(f"Alert: {alert_type} - {message}")

    def _get_severity(self, alert_type: str) -> str:
        """获取告警严重级别"""
        severity_map = {
            'HIGH_CPU_USAGE': 'warning',
            'HIGH_MEMORY_USAGE': 'warning',
            'HIGH_ERROR_RATE': 'critical',
            'HIGH_RESPONSE_TIME': 'warning',
            'NODE_FAILURE': 'critical'
        }
        return severity_map.get(alert_type, 'info')

    async def _send_notification(self, alert: Dict[str, Any]):
        """发送告警通知"""
        # 这里可以实现各种通知方式
        # 例如：邮件、短信、Slack、PagerDuty等

        # 简化实现，只记录日志
        self.logger.info(f"Sending notification for alert: {alert['type']}")
```

## 实际应用案例

### 构建分布式缓存系统

```python
class DistributedCache:
    """分布式缓存系统"""

    def __init__(self, nodes: List[str],
                 sharding_strategy: ShardingStrategy = None,
                 replication_factor: int = 3):
        self.nodes = nodes
        self.sharding_strategy = sharding_strategy or HashSharding()
        self.replication_factor = replication_factor
        self.data_store = DistributedDataStore(nodes, sharding_strategy, replication_factor)
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger(__name__)

    async def get(self, key: str) -> Any:
        """获取缓存值"""
        start_time = time.time()

        try:
            value = await self.data_store.get(key)

            # 记录指标
            self.metrics_collector.record_metric(
                "cache.get",
                1 if value is not None else 0,
                {"hit": str(value is not None)}
            )

            # 记录延迟
            latency = time.time() - start_time
            self.metrics_collector.record_metric("cache.get_latency", latency)

            return value

        except Exception as e:
            self.logger.error(f"Error getting key {key}: {e}")
            self.metrics_collector.record_metric("cache.get_error", 1)
            return None

    async def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        start_time = time.time()

        try:
            # 如果有TTL，包装值
            if ttl:
                value = {
                    'data': value,
                    'expire_at': time.time() + ttl
                }

            success = await self.data_store.put(key, value)

            # 记录指标
            self.metrics_collector.record_metric(
                "cache.put",
                1 if success else 0
            )

            # 记录延迟
            latency = time.time() - start_time
            self.metrics_collector.record_metric("cache.put_latency", latency)

            return success

        except Exception as e:
            self.logger.error(f"Error putting key {key}: {e}")
            self.metrics_collector.record_metric("cache.put_error", 1)
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            # 简化实现，使用put来删除
            return await self.data_store.put(key, None)
        except Exception as e:
            self.logger.error(f"Error deleting key {key}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {}

        # 计算命中率
        get_hits = self.metrics_collector.get_average(
            "cache.get", 60, {"hit": "True"}
        )
        get_misses = self.metrics_collector.get_average(
            "cache.get", 60, {"hit": "False"}
        )

        total_requests = get_hits + get_misses
        hit_rate = get_hits / total_requests if total_requests > 0 else 0

        stats['hit_rate'] = hit_rate
        stats['total_requests'] = total_requests

        # 延迟统计
        stats['avg_get_latency'] = self.metrics_collector.get_average(
            "cache.get_latency", 60
        )
        stats['avg_put_latency'] = self.metrics_collector.get_average(
            "cache.put_latency", 60
        )

        # 错误率
        get_errors = self.metrics_collector.get_average("cache.get_error", 60)
        put_errors = self.metrics_collector.get_average("cache.put_error", 60)

        stats['error_rate'] = (get_errors + put_errors) / max(total_requests, 1)

        return stats

class CacheWarmer:
    """缓存预热器"""

    def __init__(self, cache: DistributedCache,
                 warm_up_script: str = None):
        self.cache = cache
        self.warm_up_script = warm_up_script
        self.logger = logging.getLogger(__name__)

    async def warm_up(self, keys: List[str], data_source: callable):
        """预热缓存"""
        self.logger.info(f"Starting cache warm-up for {len(keys)} keys")

        # 分批预热
        batch_size = 100
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]

            tasks = []
            for key in batch:
                task = asyncio.create_task(self._warm_up_key(key, data_source))
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

            self.logger.info(f"Warmed up batch {i//batch_size + 1}/{(len(keys)-1)//batch_size + 1}")

        self.logger.info("Cache warm-up completed")

    async def _warm_up_key(self, key: str, data_source: callable):
        """预热单个键"""
        try:
            # 检查是否已经在缓存中
            existing_value = await self.cache.get(key)
            if existing_value is not None:
                return

            # 从数据源获取数据
            value = await data_source(key)
            if value is not None:
                await self.cache.put(key, value)

        except Exception as e:
            self.logger.error(f"Failed to warm up key {key}: {e}")
```

## 性能优化与最佳实践

### 性能调优策略

```python
class PerformanceOptimizer:
    """性能优化器"""

    def __init__(self, system: DistributedCache):
        self.system = system
        self.logger = logging.getLogger(__name__)

    async def optimize_sharding(self) -> Dict[str, Any]:
        """优化分片策略"""
        self.logger.info("Starting sharding optimization")

        # 分析当前分片负载
        shard_loads = {}
        for i, node in enumerate(self.system.nodes):
            load = await self._measure_node_load(node)
            shard_loads[i] = load

        # 计算负载均衡指标
        avg_load = sum(shard_loads.values()) / len(shard_loads)
        load_variance = sum((load - avg_load) ** 2 for load in shard_loads.values()) / len(shard_loads)

        # 如果负载不均衡，建议重新分片
        recommendations = []

        if load_variance > avg_load * 0.5:  # 标准差超过平均值的50%
            recommendations.append("考虑重新平衡分片")

            # 建议新的分片策略
            if isinstance(self.system.sharding_strategy, HashSharding):
                recommendations.append("考虑使用一致性哈希减少重新分片开销")

        return {
            'current_loads': shard_loads,
            'average_load': avg_load,
            'load_variance': load_variance,
            'recommendations': recommendations
        }

    async def optimize_replication(self) -> Dict[str, Any]:
        """优化复制策略"""
        self.logger.info("Starting replication optimization")

        # 分析复制延迟
        replication_lags = {}
        for i, node in enumerate(self.system.nodes):
            lag = await self._measure_replication_lag(node)
            replication_lags[node] = lag

        # 分析可用性
        availability = {}
        for node in self.system.nodes:
            availability[node] = await self._measure_node_availability(node)

        # 生成优化建议
        recommendations = []

        # 检查是否有节点的复制延迟过高
        high_lag_nodes = [node for node, lag in replication_lags.items() if lag > 1.0]
        if high_lag_nodes:
            recommendations.append(f"节点 {high_lag_nodes} 复制延迟过高，需要检查网络连接")

        # 检查可用性
        low_availability_nodes = [
            node for node, avail in availability.items()
            if avail < 0.99
        ]
        if low_availability_nodes:
            recommendations.append(f"节点 {low_availability_nodes} 可用性较低，考虑增加副本")

        return {
            'replication_lags': replication_lags,
            'availability': availability,
            'recommendations': recommendations
        }

    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """优化内存使用"""
        self.logger.info("Starting memory optimization")

        # 分析内存使用模式
        memory_stats = {}
        for node in self.system.nodes:
            stats = await self._analyze_memory_usage(node)
            memory_stats[node] = stats

        # 生成优化建议
        recommendations = []

        # 检查内存压力
        high_memory_nodes = [
            node for node, stats in memory_stats.items()
            if stats.get('usage_percent', 0) > 80
        ]
        if high_memory_nodes:
            recommendations.append(f"节点 {high_memory_nodes} 内存使用率过高")
            recommendations.append("考虑增加节点或实施内存清理策略")

        # 检查缓存命中率
        low_hit_rate_nodes = [
            node for node, stats in memory_stats.items()
            if stats.get('hit_rate', 0) < 0.7
        ]
        if low_hit_rate_nodes:
            recommendations.append(f"节点 {low_hit_rate_nodes} 缓存命中率较低")
            recommendations.append("考虑调整缓存策略或增加缓存大小")

        return {
            'memory_stats': memory_stats,
            'recommendations': recommendations
        }

    async def _measure_node_load(self, node: str) -> float:
        """测量节点负载"""
        try:
            # 模拟负载测量
            await asyncio.sleep(0.1)
            return random.uniform(0.3, 0.9)
        except Exception as e:
            self.logger.error(f"Failed to measure load for {node}: {e}")
            return 0.0

    async def _measure_replication_lag(self, node: str) -> float:
        """测量复制延迟"""
        try:
            await asyncio.sleep(0.1)
            return random.uniform(0.01, 2.0)
        except Exception as e:
            self.logger.error(f"Failed to measure replication lag for {node}: {e}")
            return 0.0

    async def _measure_node_availability(self, node: str) -> float:
        """测量节点可用性"""
        try:
            await asyncio.sleep(0.1)
            return random.uniform(0.95, 0.999)
        except Exception as e:
            self.logger.error(f"Failed to measure availability for {node}: {e}")
            return 0.0

    async def _analyze_memory_usage(self, node: str) -> Dict[str, Any]:
        """分析内存使用"""
        try:
            await asyncio.sleep(0.1)
            return {
                'usage_percent': random.uniform(40, 95),
                'hit_rate': random.uniform(0.6, 0.95),
                'eviction_rate': random.uniform(0.01, 0.1),
                'fragmentation': random.uniform(0.1, 0.3)
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze memory usage for {node}: {e}")
            return {}
```

## 总结与展望

分布式系统设计是一个复杂而深刻的主题，它涉及到计算机科学的多个领域。通过本文的学习，我们深入探讨了：

### 核心概念
- **CAP定理**：理解一致性、可用性和分区容忍性的权衡
- **一致性模型**：从强一致性到最终一致性的完整谱系
- **共识算法**：Raft和Paxos的深入实现

### 实践技能
- **分片策略**：哈希分片、范围分片和一致性哈希
- **复制管理**：主从复制、多主复制和仲裁写入
- **分布式事务**：两阶段提交和Saga模式

### 高级主题
- **故障检测**：心跳机制和健康检查
- **系统监控**：指标收集和告警管理
- **性能优化**：负载均衡和资源管理

### 未来发展
随着云原生技术的不断发展，分布式系统正在向着更加智能化、自动化的方向发展。服务网格、无服务器架构和边缘计算等新技术正在重塑分布式系统的设计理念。

**记住**：优秀的分布式系统设计不仅仅是技术实现，更是对业务需求的深刻理解和对系统复杂性的优雅控制。

---

*这篇博客涵盖了Python分布式系统设计的核心概念和实际实现。通过深入的理论分析和丰富的代码示例，你可以掌握构建高可用、高性能分布式系统的关键技术。*