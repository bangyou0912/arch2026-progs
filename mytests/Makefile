# Worst-case rubik measurement on rv32emu, for every optimization level.
# Run from the repository root after `make elfs`: make -f mytests/Makefile
include mk/toolchain.mk
BUILD := build
OPTS := O1 O2 O3 Os
EMU := third_party/rv32emu/build/rv32emu
COMMON_OBJS := $(addprefix $(BUILD)/common/, start.o perf.o rt.o print.o)
CFLAGS := -march=rv32i_zicsr -mabi=ilp32 -ffreestanding -fno-pic \
          -fno-stack-protector -nostdlib -iquote . -Icommon -O2

all: $(foreach o,$(OPTS),run-$(o))

$(BUILD)/mytests/worst.o: mytests/rubik_worst.c mytests/rubik_states.h
	mkdir -p $(@D)
	$(CROSS_COMPILE)gcc $(CFLAGS) -c $< -o $@

$(BUILD)/mytests/worst-%.elf: $(BUILD)/mytests/worst.o $(BUILD)/rubik/%/impl.o $(BUILD)/rubik/impl_asm.o
	$(CROSS_COMPILE)ld -T common/linker.ld -o $@ $^ $(COMMON_OBJS)

run-%: $(BUILD)/mytests/worst-%.elf
	@echo "== -$*"; $(EMU) $< | grep -E "@@worst|MISMATCH"
