using Test

include("../src/hllset_swarm/HllSets.jl")
using .HllSets

@testset "HllSets Tests" begin
    
    @testset "Basic Construction" begin
        hll = HllSet{10}()
        @test length(hll.counts) == 1024
        @test all(hll.counts .== 0)
        @test isempty(hll)
        
        # Test different precision values
        @test length(HllSet{4}().counts) == 16
        @test length(HllSet{12}().counts) == 4096
        @test length(HllSet{18}().counts) == 262144
    end

    @testset "Invalid Construction" begin
        # Test invalid P values
        @test_throws ArgumentError HllSet{3}()
        @test_throws ArgumentError HllSet{19}()
        @test_throws ArgumentError HllSet{0}()
    end

    @testset "Add Single Element" begin
        hll = HllSet{10}()
        
        result = add!(hll, "test")
        @test length(result) == 4  # (element, hash, bin, idx)
        @test result[1] == "test"
        @test result[2] > 0  # hash value
        @test 1 <= result[3] <= 1024  # bin
        @test result[4] > 0  # idx
        
        @test !isempty(hll)
        @test any(hll.counts .!= 0)
    end

    @testset "Add Multiple Elements" begin
        hll = HllSet{10}()
        
        elements = ["a", "b", "c", "d", "e"]
        results = add!(hll, elements)
        
        @test length(results) == 5
        @test all(length.(results) .== 4)
        @test !isempty(hll)
    end

    @testset "Add with Seed" begin
        hll1 = HllSet{10}()
        hll2 = HllSet{10}()
        
        add!(hll1, "test", seed=0)
        add!(hll2, "test", seed=42)
        
        # Same element with different seeds should produce different results
        @test !isequal(hll1, hll2)
    end

    @testset "Cardinality Estimation" begin
        hll = HllSet{10}()
        
        # Add known number of unique elements
        n_elements = 1000
        for i in 1:n_elements
            add!(hll, "element_$i")
        end
        
        estimated = count(hll)
        error_rate = abs(estimated - n_elements) / n_elements
        
        @test estimated > 0
        @test error_rate < 0.05  # HLL typically has ~2% error for P=10
        println("Cardinality Test: Added $n_elements, Estimated $estimated, Error $(round(error_rate * 100, digits=2))%")
    end

    @testset "Cardinality with Duplicates" begin
        hll = HllSet{10}()
        
        # Add same elements multiple times
        for _ in 1:10
            for i in 1:100
                add!(hll, "element_$i")
            end
        end
        
        estimated = count(hll)
        error_rate = abs(estimated - 100) / 100
        
        @test error_rate < 0.1  # Should still estimate ~100 unique elements
        println("Duplicates Test: Expected 100, Estimated $estimated, Error $(round(error_rate * 100, digits=2))%")
    end

    @testset "Union Operations" begin
        hll1 = HllSet{10}()
        hll2 = HllSet{10}()
        
        add!(hll1, ["a", "b", "c"])
        add!(hll2, ["c", "d", "e"])
        
        hll_union = HllSets.union(hll1, hll2)
        
        @test count(hll_union) >= max(count(hll1), count(hll2))
        @test count(hll_union) <= count(hll1) + count(hll2)
        
        # Test in-place union
        hll3 = HllSet{10}()
        add!(hll3, ["a", "b"])
        union!(hll3, hll1)
        @test count(hll3) >= count(hll1)
    end

    @testset "Intersection Operations" begin
        hll1 = HllSet{10}()
        hll2 = HllSet{10}()
        
        # Add overlapping elements
        shared = ["shared1", "shared2", "shared3"]
        add!(hll1, vcat(shared, ["unique1", "unique2"]))
        add!(hll2, vcat(shared, ["unique3", "unique4"]))
        
        hll_inter = HllSets.intersect(hll1, hll2)
        
        @test count(hll_inter) > 0
        @test count(hll_inter) <= min(count(hll1), count(hll2))
    end

    @testset "Difference Operations" begin
        hll1 = HllSet{10}()
        hll2 = HllSet{10}()
        
        add!(hll1, ["a", "b", "c", "d"])
        add!(hll2, ["c", "d", "e", "f"])
        
        result = diff(hll1, hll2)
        
        @test haskey(result, :DEL)
        @test haskey(result, :RET)
        @test haskey(result, :NEW)
        
        @test count(result.DEL) >= 0  # Elements in hll1 not in hll2
        @test count(result.NEW) >= 0  # Elements in hll2 not in hll1
        @test count(result.RET) >= 0  # Common elements
    end

    @testset "Complement Operations" begin
        hll1 = HllSet{10}()
        hll2 = HllSet{10}()
        
        add!(hll1, ["a", "b", "c", "d", "e"])
        add!(hll2, ["b", "c"])
        
        comp = set_comp(hll1, hll2)
        
        @test count(comp) > 0
        @test count(comp) <= count(hll1)
    end

    @testset "XOR Operations" begin
        hll1 = HllSet{10}()
        hll2 = HllSet{10}()
        
        add!(hll1, ["a", "b", "c"])
        add!(hll2, ["b", "c", "d"])
        
        xor_result = set_xor(hll1, hll2)
        @test count(xor_result) >= 0
    end

    @testset "Copy Operations" begin
        hll1 = HllSet{10}()
        add!(hll1, ["a", "b", "c", "d", "e"])
        
        # Test copy! with destination
        hll2 = HllSet{10}()
        copy!(hll2, hll1)
        @test isequal(hll1, hll2)
        
        # Test copy! without destination
        hll3 = copy!(hll1)
        @test isequal(hll1, hll3)
        @test hll1.counts !== hll3.counts  # Different objects
        
        # Modify hll3 and check hll1 is unchanged
        add!(hll3, "new_element")
        @test !isequal(hll1, hll3)
    end

    @testset "Equality Operations" begin
        hll1 = HllSet{10}()
        hll2 = HllSet{10}()
        hll3 = HllSet{10}()
        
        # Empty sets should be equal
        @test isequal(hll1, hll2)
        
        # Same elements should produce equal sets
        add!(hll1, ["a", "b", "c"])
        add!(hll2, ["a", "b", "c"])
        @test isequal(hll1, hll2)
        
        # Different elements should not be equal
        add!(hll3, ["d", "e", "f"])
        @test !isequal(hll1, hll3)
        
        # Test with incompatible sizes
        hll4 = HllSet{12}()
        @test_throws ArgumentError isequal(hll1, hll4)
    end

    @testset "Jaccard Similarity (Match)" begin
        hll1 = HllSet{10}()
        hll2 = HllSet{10}()
        
        # Identical sets should have ~100% similarity
        add!(hll1, ["a", "b", "c", "d", "e"])
        add!(hll2, ["a", "b", "c", "d", "e"])
        similarity = match(hll1, hll2)
        @test similarity > 90
        println("Match Test (identical): $(similarity)%")
        
        # Completely different sets should have low similarity
        hll3 = HllSet{10}()
        hll4 = HllSet{10}()
        add!(hll3, ["a", "b", "c"])
        add!(hll4, ["x", "y", "z"])
        similarity2 = match(hll3, hll4)
        @test similarity2 < 20
        println("Match Test (different): $(similarity2)%")
        
        # Partially overlapping sets
        hll5 = HllSet{10}()
        hll6 = HllSet{10}()
        add!(hll5, ["a", "b", "c", "d"])
        add!(hll6, ["c", "d", "e", "f"])
        similarity3 = match(hll5, hll6)
        @test 15 < similarity3 < 70
        println("Match Test (partial): $(similarity3)%")
    end

    @testset "Cosine Similarity" begin
        hll1 = HllSet{10}()
        hll2 = HllSet{10}()
        
        add!(hll1, ["a", "b", "c", "d"])
        add!(hll2, ["a", "b", "e", "f"])
        
        sim = cosine(hll1, hll2)
        @test 0.0 <= sim <= 1.0
        println("Cosine Similarity: $(round(sim, digits=4))")
        
        # Identical sets should have cosine similarity close to 1
        hll3 = HllSet{10}()
        hll4 = HllSet{10}()
        add!(hll3, ["a", "b", "c"])
        add!(hll4, ["a", "b", "c"])
        sim2 = cosine(hll3, hll4)
        @test sim2 > 0.99
    end

    @testset "Binary Tensor Conversion" begin
        hll = HllSet{10}()
        add!(hll, ["a", "b", "c", "d", "e"])
        
        # Convert to binary tensor
        tensor = to_binary_tensor(hll)
        @test size(tensor) == (1024, 32)
        @test eltype(tensor) == Bool
        
        # Flatten tensor
        flattened = flatten_tensor(tensor)
        @test length(flattened) == 1024 * 32
        
        # Convert to string
        str = tensor_to_string(flattened)
        @test length(str) == 1024 * 32
        @test all(c in ['0', '1'] for c in str)
        
        # Convert back to tensor
        restored_tensor = string_to_tensor(str, 10)
        @test size(restored_tensor) == size(tensor)
        @test restored_tensor == tensor
        
        # Convert back to HllSet
        restored_hll = binary_tensor_to_hllset(restored_tensor, 10)
        @test isequal(hll, restored_hll)
    end

    @testset "ID Generation" begin
        hll1 = HllSet{10}()
        hll2 = HllSet{10}()
        
        add!(hll1, ["a", "b", "c"])
        add!(hll2, ["a", "b", "c"])
        
        id1 = id(hll1)
        id2 = id(hll2)
        
        @test typeof(id1) == String
        @test length(id1) == 40  # SHA1 produces 40 hex characters
        @test id1 == id2  # Same content should produce same ID
        
        # Different content should produce different ID
        hll3 = HllSet{10}()
        add!(hll3, ["x", "y", "z"])
        id3 = id(hll3)
        @test id1 != id3
        
        # Empty set should have consistent ID
        empty_hll1 = HllSet{10}()
        empty_hll2 = HllSet{10}()
        @test id(empty_hll1) == id(empty_hll2)
    end

    @testset "Hash Function (u_hash)" begin
        # Test basic hashing
        h1 = u_hash("test")
        @test h1 > 0
        @test h1 isa Int
        
        # Same input should produce same hash
        h2 = u_hash("test")
        @test h1 == h2
        
        # Different inputs should (likely) produce different hashes
        h3 = u_hash("different")
        @test h1 != h3
        
        # Test with seed
        h4 = u_hash("test", seed=42)
        @test h4 != h1
        
        # Same input with same seed should be consistent
        h5 = u_hash("test", seed=42)
        @test h4 == h5
    end

    @testset "Helper Functions" begin
        hll = HllSet{10}()
        
        # Test getbin
        h = u_hash("test")
        bin = getbin(hll, h)
        @test 1 <= bin <= 1024
        
        # Test getzeros
        zeros_count = getzeros(hll, h)
        @test zeros_count > 0
        
        # Test maxidx
        idx = maxidx(UInt32(15))
        @test idx > 0
    end

    @testset "Sizeof Operations" begin
        @test sizeof(HllSet{10}) == 1024
        @test sizeof(HllSet{12}) == 4096
        
        hll = HllSet{10}()
        @test sizeof(hll) == 1024
    end

    @testset "Incompatible Operations" begin
        hll10 = HllSet{10}()
        hll12 = HllSet{12}()
        
        add!(hll10, ["a", "b"])
        add!(hll12, ["c", "d"])
        
        # These should throw ArgumentError for mismatched precision
        @test_throws ArgumentError HllSets.union(hll10, hll12)
        @test_throws ArgumentError HllSets.intersect(hll10, hll12)
        @test_throws ArgumentError diff(hll10, hll12)
        @test_throws ArgumentError set_comp(hll10, hll12)
        @test_throws ArgumentError set_xor(hll10, hll12)
        @test_throws ArgumentError match(hll10, hll12)
        @test_throws ArgumentError cosine(hll10, hll12)
        @test_throws ArgumentError copy!(hll10, hll12)
        @test_throws ArgumentError isequal(hll10, hll12)
    end

    @testset "Large Scale Test" begin
        hll = HllSet{12}()
        
        n = 10000
        for i in 1:n
            add!(hll, "item_$i")
        end
        
        estimated = count(hll)
        error = abs(estimated - n) / n
        
        println("\nLarge scale test (P=12, $n elements):")
        println("  Estimated: $estimated")
        println("  Error: $(round(error * 100, digits=2))%")
        
        @test error < 0.05
    end

    @testset "Very Large Scale Test" begin
        hll = HllSet{14}()
        
        n = 100000
        for i in 1:n
            add!(hll, "item_$i")
        end
        
        estimated = count(hll)
        error = abs(estimated - n) / n
        
        println("\nVery large scale test (P=14, $n elements):")
        println("  Estimated: $estimated")
        println("  Error: $(round(error * 100, digits=2))%")
        
        @test error < 0.03  # Better precision with P=14
    end

    @testset "Edge Cases" begin
        hll = HllSet{10}()
        
        # Empty set operations
        empty1 = HllSet{10}()
        empty2 = HllSet{10}()
        
        @test isempty(empty1)
        @test count(empty1) >= 0
        # @test isequal(empty1, empty2)
        
        union_empty = HllSets.union(empty1, empty2)
        @test isempty(union_empty)
        
        # Add and check not empty
        add!(hll, "single")
        @test !isempty(hll)
        
        # Union with empty
        result = HllSets.union(hll, empty1)
        @test count(result) == count(hll)
    end

    @testset "String vs Number Hashing" begin
        hll_str = HllSet{10}()
        hll_num = HllSet{10}()
        
        # Add strings
        for i in 1:100
            add!(hll_str, "item_$i")
        end
        
        # Add numbers
        for i in 1:100
            add!(hll_num, i)
        end
        
        # Both should estimate ~100 elements
        @test 90 < count(hll_str) < 110
        @test 90 < count(hll_num) < 110
        
        # But should have different IDs
        @test id(hll_str) != id(hll_num)
    end

end

println("\n✓ All HllSets tests passed!")